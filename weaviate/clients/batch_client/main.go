package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/kshedden/gonpy"

	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/filters"
	wgraphql "github.com/weaviate/weaviate-go-client/v5/weaviate/graphql"
	wgrpc "github.com/weaviate/weaviate-go-client/v5/weaviate/grpc"
	"github.com/weaviate/weaviate/entities/models"
)

const idPropName = "doc_id"
const searchableStabilizationPeriod = 30 * time.Second

type NodeInfo struct {
	Rank     int
	Node     string
	IP       string
	HTTPPort int
	GRPCPort int
}

type SweepConfig struct {
	BatchSize  int
	ResultPath string
	Label      string
}

type NpyMetadata struct {
	DataOffset int64
	Rows       int
	Cols       int
}

type InputData struct {
	Streaming bool
	Matrix    [][]float32
	Meta      *NpyMetadata
	Path      string
}

type collectionSnapshot struct {
	TotalQueue   int
	Shards       int
	Indexing     int
	TotalObjects int
	ShardRefs    []ShardRef
}

type ShardRef struct {
	NodeName    string
	ShardName   string
	ObjectCount int
}

var debugHTTPClient = &http.Client{
	Timeout: 9999 * time.Second,
}

// parseNpyMetadata reads only the .npy header so streaming mode can inspect
// matrix shape without loading the full file.
func parseNpyMetadata(path string) (*NpyMetadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	magic := make([]byte, 6)
	if _, err := io.ReadFull(file, magic); err != nil {
		return nil, err
	}
	if string(magic) != "\x93NUMPY" {
		return nil, fmt.Errorf("%s is not a valid .npy file", path)
	}

	version := make([]byte, 2)
	if _, err := io.ReadFull(file, version); err != nil {
		return nil, err
	}

	var headerLen int
	switch version[0] {
	case 1:
		var length uint16
		if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		headerLen = int(length)
	case 2, 3:
		var length uint32
		if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		headerLen = int(length)
	default:
		return nil, fmt.Errorf("unsupported npy version %d.%d", version[0], version[1])
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, err
	}
	header := strings.TrimSpace(string(headerBytes))

	descr, err := parseNpyHeaderString(header, "descr")
	if err != nil {
		return nil, err
	}
	if descr != "<f4" && descr != "=f4" && descr != "f4" {
		return nil, fmt.Errorf("unsupported npy dtype %q; expected float32", descr)
	}

	fortranOrder, err := parseNpyHeaderBool(header, "fortran_order")
	if err != nil {
		return nil, err
	}
	if fortranOrder {
		return nil, fmt.Errorf("unsupported npy layout; only C-order arrays are supported")
	}

	shapeRaw, err := parseNpyHeaderTuple(header, "shape")
	if err != nil {
		return nil, err
	}
	parts := strings.Split(shapeRaw, ",")
	dims := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid npy shape component %q: %w", part, err)
		}
		dims = append(dims, value)
	}
	if len(dims) != 2 {
		return nil, fmt.Errorf("expected a 2D npy array, got shape %v", dims)
	}

	offset, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	return &NpyMetadata{
		DataOffset: offset,
		Rows:       dims[0],
		Cols:       dims[1],
	}, nil
}

// parseNpyHeaderString extracts one quoted string field from the NumPy header.
func parseNpyHeaderString(header, key string) (string, error) {
	needle := fmt.Sprintf("'%s'", key)
	start := strings.Index(header, needle)
	if start < 0 {
		return "", fmt.Errorf("npy header missing %q", key)
	}
	rest := header[start+len(needle):]
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return "", fmt.Errorf("npy header missing ':' after %q", key)
	}
	rest = strings.TrimSpace(rest[colon+1:])
	if len(rest) == 0 {
		return "", fmt.Errorf("npy header missing value for %q", key)
	}

	quote := rest[0]
	if quote != '\'' && quote != '"' {
		return "", fmt.Errorf("npy header value for %q was not quoted", key)
	}
	rest = rest[1:]
	end := strings.IndexByte(rest, quote)
	if end < 0 {
		return "", fmt.Errorf("unterminated string value for %q", key)
	}
	return rest[:end], nil
}

// parseNpyHeaderTuple extracts the tuple payload used for shape parsing.
func parseNpyHeaderTuple(header, key string) (string, error) {
	needle := fmt.Sprintf("'%s'", key)
	start := strings.Index(header, needle)
	if start < 0 {
		return "", fmt.Errorf("npy header missing %q", key)
	}
	rest := header[start+len(needle):]
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return "", fmt.Errorf("npy header missing ':' after %q", key)
	}
	rest = strings.TrimSpace(rest[colon+1:])
	open := strings.Index(rest, "(")
	if open < 0 {
		return "", fmt.Errorf("npy header missing tuple for %q", key)
	}
	rest = rest[open+1:]
	closeIdx := strings.Index(rest, ")")
	if closeIdx < 0 {
		return "", fmt.Errorf("npy header missing ')' for %q", key)
	}
	return rest[:closeIdx], nil
}

// parseNpyHeaderBool extracts a boolean field from the NumPy header.
func parseNpyHeaderBool(header, key string) (bool, error) {
	needle := fmt.Sprintf("'%s'", key)
	start := strings.Index(header, needle)
	if start < 0 {
		return false, fmt.Errorf("npy header missing %q", key)
	}
	rest := header[start+len(needle):]
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return false, fmt.Errorf("npy header missing ':' after %q", key)
	}
	rest = strings.TrimSpace(rest[colon+1:])
	if strings.HasPrefix(rest, "True") {
		return true, nil
	}
	if strings.HasPrefix(rest, "False") {
		return false, nil
	}
	return false, fmt.Errorf("npy header %q value was not a boolean", key)
}

// readRowsFromNpy loads one contiguous row range from a float32 .npy matrix.
func readRowsFromNpy(file *os.File, meta *NpyMetadata, startRow, rowCount int) ([][]float32, error) {
	if rowCount == 0 {
		return nil, nil
	}

	bytesPerRow := meta.Cols * 4
	byteOffset := int64(startRow * bytesPerRow)
	offset := meta.DataOffset + byteOffset
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}

	buf := make([]byte, rowCount*bytesPerRow)
	if _, err := io.ReadFull(file, buf); err != nil {
		return nil, err
	}

	data := make([]float32, rowCount*meta.Cols)
	for i := range data {
		base := i * 4
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[base : base+4]))
	}

	rows := make([][]float32, rowCount)
	for i := 0; i < rowCount; i++ {
		start := i * meta.Cols
		end := start + meta.Cols
		rows[i] = data[start:end]
	}
	return rows, nil
}

// runtimeStatePath keeps workflow marker files under the generated runtime
// state directory expected by the surrounding shell harness.
func runtimeStatePath(name string) string {
	dir := strings.TrimSpace(os.Getenv("RUNTIME_STATE_DIR"))
	if dir == "" {
		dir = "./runtime_state"
	}
	return dir + "/" + name
}

func writeTextFile(path string, contents string) error {
	return os.WriteFile(path, []byte(contents), 0644)
}

// writeInt64Npy preserves the existing query-result artifact format used by
// the other batch clients and downstream analysis scripts.
func writeInt64Npy(path string, data []int64, shape []int) error {
	w, err := gonpy.NewFileWriter(path)
	if err != nil {
		return err
	}
	if shape != nil {
		w.Shape = shape
	}
	return w.WriteInt64(data)
}

// getNodeByRank parses the Weaviate registry and selects the target node used
// for worker-based balancing.
func getNodeByRank(filename string, targetRank int) (*NodeInfo, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}

		rank, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		if rank != targetRank {
			continue
		}

		httpPort, err := strconv.Atoi(strings.TrimSpace(parts[3]))
		if err != nil {
			return nil, err
		}
		grpcPort, err := strconv.Atoi(strings.TrimSpace(parts[4]))
		if err != nil {
			return nil, err
		}

		return &NodeInfo{
			Rank:     rank,
			Node:     strings.TrimSpace(parts[1]),
			IP:       strings.TrimSpace(parts[2]),
			HTTPPort: httpPort,
			GRPCPort: grpcPort,
		}, nil
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return nil, fmt.Errorf("rank %d not found in %s", targetRank, filename)
}

func getAllNodes(filename string) ([]*NodeInfo, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var nodes []*NodeInfo
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}

		rank, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		httpPort, err := strconv.Atoi(strings.TrimSpace(parts[3]))
		if err != nil {
			continue
		}
		grpcPort, err := strconv.Atoi(strings.TrimSpace(parts[4]))
		if err != nil {
			continue
		}

		nodes = append(nodes, &NodeInfo{
			Rank:     rank,
			Node:     strings.TrimSpace(parts[1]),
			IP:       strings.TrimSpace(parts[2]),
			HTTPPort: httpPort,
			GRPCPort: grpcPort,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes found in %s", filename)
	}
	return nodes, nil
}

// getCollectionSnapshot summarizes collection state from the official cluster
// nodes API so rank 0 can decide when inserts are fully visible and indexing
// has drained.
func getCollectionSnapshot(ctx context.Context, client *weaviate.Client, className string) (collectionSnapshot, error) {
	parsed, err := client.Cluster().
		NodesStatusGetter().
		WithClass(className).
		WithOutput("verbose").
		Do(ctx)
	if err != nil {
		return collectionSnapshot{}, err
	}

	snap := collectionSnapshot{}
	for _, nodeResp := range parsed.Nodes {
		for _, shard := range nodeResp.Shards {
			if shard.Class != className {
				continue
			}
			snap.Shards++
			snap.TotalObjects += int(shard.ObjectCount)
			snap.TotalQueue += int(shard.VectorQueueLength)
			snap.ShardRefs = append(snap.ShardRefs, ShardRef{
				NodeName:    nodeResp.Name,
				ShardName:   shard.Name,
				ObjectCount: int(shard.ObjectCount),
			})
			if strings.EqualFold(shard.VectorIndexingStatus, "INDEXING") {
				snap.Indexing++
			}
		}
	}
	return snap, nil
}

func getUnderlyingIndexStatus(ctx context.Context, nodes []*NodeInfo, snap collectionSnapshot, className string) (bool, string, error) {
	if len(snap.ShardRefs) == 0 {
		return false, "", fmt.Errorf("no shard refs found for collection %s", className)
	}

	nodeByName := make(map[string]*NodeInfo, len(nodes))
	for _, node := range nodes {
		nodeByName[node.Node] = node
	}

	var details []string
	targetVector := "default"
	hnswCount := 0
	flatCount := 0

	for _, shardRef := range snap.ShardRefs {
		node := nodeByName[shardRef.NodeName]
		if node == nil {
			if len(nodes) == 1 {
				node = nodes[0]
			} else {
				return false, "", fmt.Errorf("no node mapping found for shard node %s", shardRef.NodeName)
			}
		}

		debugURL := fmt.Sprintf(
			"http://%s:%d/debug/stats/collection/%s/shards/%s/%s",
			node.IP,
			node.HTTPPort+1,
			url.PathEscape(className),
			url.PathEscape(shardRef.ShardName),
			url.PathEscape(targetVector),
		)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, debugURL, nil)
		if err != nil {
			return false, "", err
		}

		resp, err := debugHTTPClient.Do(req)
		if err != nil {
			return false, "", err
		}

		body, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil {
			return false, "", fmt.Errorf("read debug stats from %s: %w", debugURL, readErr)
		}
		if closeErr != nil {
			return false, "", closeErr
		}

		snippet := strings.TrimSpace(string(body))
		if len(snippet) > 256 {
			snippet = snippet[:256]
		}
		switch resp.StatusCode {
		case http.StatusOK:
			hnswCount++
			details = append(details, fmt.Sprintf(
				"%s/%s objects=%d vector=%s status=200 body=%q",
				node.IP,
				shardRef.ShardName,
				shardRef.ObjectCount,
				targetVector,
				snippet,
			))
		case http.StatusBadRequest:
			flatCount++
			details = append(details, fmt.Sprintf(
				"%s/%s objects=%d vector=%s status=400 body=%q",
				node.IP,
				shardRef.ShardName,
				shardRef.ObjectCount,
				targetVector,
				snippet,
			))
		default:
			return false, "", fmt.Errorf("debug stats %s returned http %d body=%q", debugURL, resp.StatusCode, snippet)
		}
	}

	summary := fmt.Sprintf(
		"shards=%d hnsw_indexes=%d flat_indexes=%d",
		len(snap.ShardRefs),
		hnswCount,
		flatCount,
	)
	if len(details) == 0 {
		return flatCount == 0, summary, nil
	}

	return flatCount == 0, summary + "; " + strings.Join(details, "; "), nil
}

// waitForQuiescence is the Weaviate-native completion check.
// It waits for indexing queue quiescence to remain stable long enough to
// declare the collection searchable.
func waitForQuiescence(
	ctx context.Context,
	client *weaviate.Client,
	nodes []*NodeInfo,
	className string,
	requireHNSW bool,
	pollInterval time.Duration,
) (time.Time, error) {
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}

	debugEnabled := envEnabled("DEBUG")
	var last collectionSnapshot
	lastHNSW := !requireHNSW
	lastHNSWDetail := "not required"
	lastWaitLog := time.Time{}
	currentObjects := last.TotalObjects
	currentQueue := last.TotalQueue
	currentIndexing := last.Indexing
	currentHNSW := lastHNSW
	currentHNSWDetail := lastHNSWDetail
	var stableSince time.Time
	logWaitErr := func(label string, err error) {
		if err == nil {
			return
		}

		log.Printf(
			"Quiescence poll %s: %v (objects=%d queue=%d indexing=%d hnsw=%t details=%s)",
			label,
			err,
			currentObjects,
			currentQueue,
			currentIndexing,
			currentHNSW,
			currentHNSWDetail,
		)
	}

	for {
		snap, err := getCollectionSnapshot(ctx, client, className)
		if err == nil {
			last = snap
			currentObjects = snap.TotalObjects
			currentQueue = snap.TotalQueue
			currentIndexing = snap.Indexing
			hnswReady := !requireHNSW
			hnswDetail := "not required"
			currentHNSW = hnswReady
			currentHNSWDetail = hnswDetail
			if requireHNSW {
				hnswReady, hnswDetail, err = getUnderlyingIndexStatus(ctx, nodes, snap, className)
				currentHNSW = hnswReady
				currentHNSWDetail = hnswDetail
				if err != nil {
					logWaitErr("failed during HNSW readiness check", err)
				}
				if err == nil {
					lastHNSW = hnswReady
					lastHNSWDetail = hnswDetail
					currentHNSW = lastHNSW
					currentHNSWDetail = lastHNSWDetail
				}
			}
			if debugEnabled {
				log.Printf(
					"Quiescence poll: queue=%d indexing=%d hnsw=%t details=%s err=%v",
					snap.TotalQueue,
					snap.Indexing,
					hnswReady,
					hnswDetail,
					err,
				)
			}
			if err == nil && snap.TotalQueue == 0 && snap.Indexing == 0 && hnswReady {
				if stableSince.IsZero() {
					stableSince = time.Now()
					log.Printf(
						"Collection quiescent; starting %s stabilization window: queue=%d indexing=%d hnsw=%t details=%s",
						searchableStabilizationPeriod,
						snap.TotalQueue,
						snap.Indexing,
						hnswReady,
						hnswDetail,
					)
				} else if time.Since(stableSince) >= searchableStabilizationPeriod {
					return stableSince, nil
				}
			} else if !stableSince.IsZero() {
				log.Printf(
					"Collection stabilization broke after %s: queue=%d indexing=%d hnsw=%t details=%s",
					time.Since(stableSince).Round(time.Millisecond),
					snap.TotalQueue,
					snap.Indexing,
					hnswReady,
					hnswDetail,
				)
				stableSince = time.Time{}
			} else if err == nil && (lastWaitLog.IsZero() || time.Since(lastWaitLog) >= 10*time.Second) {
				log.Printf(
					"Still waiting for collection quiescence: queue=%d indexing=%d hnsw=%t details=%s",
					snap.TotalQueue,
					snap.Indexing,
					hnswReady,
					hnswDetail,
				)
				lastWaitLog = time.Now()
			}
		} else {
			logWaitErr("failed before HNSW check", err)
			if debugEnabled {
				log.Printf("Quiescence poll failed before HNSW check: err=%v", err)
			}
		}

		select {
		case <-ctx.Done():
			return time.Time{}, fmt.Errorf(
				"wait for collection quiescence failed: objects=%d queue=%d indexing=%d hnsw=%t details=%s: %w",
				last.TotalObjects,
				last.TotalQueue,
				last.Indexing,
				lastHNSW,
				lastHNSWDetail,
				ctx.Err(),
			)
		case <-time.After(pollInterval):
		}
	}
}

func logCollectionStatus(ctx context.Context, client *weaviate.Client, className string) {
	snap, err := getCollectionSnapshot(ctx, client, className)
	if err != nil {
		log.Printf("Collection status (%s): failed to fetch: %v", className, err)
		return
	}

	log.Printf(
		"Collection status (%s): objects=%d shards=%d queue=%d indexing=%d",
		className,
		snap.TotalObjects,
		snap.Shards,
		snap.TotalQueue,
		snap.Indexing,
	)
}

// queryDocIDExists does the lightweight post-run sanity check for one doc_id.
// It is only for CSV validation and is not part of completion detection.
func queryDocIDExists(ctx context.Context, client *weaviate.Client, className string, docID int64, rpcTimeout time.Duration) (bool, error) {
	queryCtx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()

	results, err := client.Experimental().Search().
		WithCollection(className).
		WithConsistencyLevel("ONE").
		WithLimit(1).
		WithProperties(idPropName).
		WithWhere(filters.Where().
			WithPath([]string{idPropName}).
			WithOperator(filters.Equal).
			WithValueInt(docID)).
		Do(queryCtx)
	if err != nil {
		return false, err
	}
	if envEnabled("DEBUG") {
		log.Printf(
			"Sanity query (%s doc_id=%d): hits=%d results=%#v",
			className,
			docID,
			len(results),
			results,
		)
	}
	return len(results) > 0, nil
}

// runOneQuery executes one nearVector gRPC search and normalizes the response
// into the same fixed-width arrays used by the prior client shape.
func runOneQuery(
	parent context.Context,
	client *weaviate.Client,
	className string,
	vec []float32,
	topK int,
	rpcTimeout time.Duration,
) (time.Duration, []int64, []float32, error) {
	ctx, cancel := context.WithTimeout(parent, rpcTimeout)
	defer cancel()

	start := time.Now()
	results, err := client.Experimental().Search().
		WithCollection(className).
		WithConsistencyLevel("ONE").
		WithLimit(topK).
		WithProperties(idPropName).
		WithMetadata(&wgraphql.Metadata{Distance: true}).
		WithNearVector(client.GraphQL().NearVectorArgBuilder().WithVector(vec)).
		Do(ctx)
	lat := time.Since(start)
	if err != nil {
		return lat, nil, nil, err
	}

	ids := make([]int64, len(results))
	dists := make([]float32, len(results))
	for i, hit := range results {
		raw, ok := hit.Properties[idPropName]
		if !ok {
			return lat, nil, nil, fmt.Errorf("query result missing %s property", idPropName)
		}
		switch value := raw.(type) {
		case int64:
			ids[i] = value
		case int:
			ids[i] = int64(value)
		case float64:
			ids[i] = int64(value)
		case float32:
			ids[i] = int64(value)
		default:
			return lat, nil, nil, fmt.Errorf("query result %s had unsupported type %T", idPropName, raw)
		}
		dists[i] = hit.Metadata.Distance
	}

	return lat, ids, dists, nil
}

// buildWeaviateBatchObjects converts one vector batch into Weaviate objects.
// This is counted as prep/construction time rather than upload time.
func buildWeaviateBatchObjects(className string, startID int, batch [][]float32) []*models.Object {
	objects := make([]*models.Object, 0, len(batch))
	for i, vec := range batch {
		objectVector := make([]float32, len(vec))
		copy(objectVector, vec)

		objects = append(objects, &models.Object{
			Class: className,
			Properties: map[string]any{
				idPropName: int64(startID + i),
			},
			Vector: objectVector,
		})
	}
	return objects
}

// Barrier represents a reusable synchronization point for a group of goroutines.
type Barrier struct {
	n     int
	count int
	gen   int
	mu    sync.Mutex
	cond  *sync.Cond
}

// Creates a reusable synchronization primitive
func NewBarrier(n int) *Barrier {
	if n <= 0 {
		panic("barrier size must be > 0")
	}
	b := &Barrier{n: n, count: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

// Wait blocks until exactly N goroutines have called it for this generation.
// Then it releases all of them and resets for the next generation.
func (b *Barrier) Wait() {
	b.mu.Lock()
	g := b.gen
	b.count--

	if b.count == 0 {
		b.gen++
		b.count = b.n
		b.cond.Broadcast()
		b.mu.Unlock()
		return
	}

	for g == b.gen {
		b.cond.Wait()
	}
	b.mu.Unlock()
}

// SharedTiming records one global window:
// loop start (first worker enters the loop) -> searchable (global visibility reached).
type SharedTiming struct {
	loopStart      time.Time
	loopDoneAt     time.Time
	searchableAt   time.Time
	startOnce      sync.Once
	loopDoneOnce   sync.Once
	searchableOnce sync.Once
	searchableCh   chan struct{}
	mu             sync.Mutex
	expected       int
	ready          int
}

// NewSharedTiming allocates the shared timing window used for the final CSV.
func NewSharedTiming(expected int) *SharedTiming {
	return &SharedTiming{
		searchableCh: make(chan struct{}),
		expected:     expected,
	}
}

// MarkLoopStart records the first client entering the active work loop.
func (t *SharedTiming) MarkLoopStart() {
	t.startOnce.Do(func() {
		t.loopStart = time.Now()
	})
}

// MarkLoopDoneAt records when all clients have finished their local work loop.
func (t *SharedTiming) MarkLoopDoneAt(at time.Time) {
	t.loopDoneOnce.Do(func() {
		t.loopDoneAt = at
	})
}

// MarkSearchable records the shared "searchable" milestone and releases
// clients waiting for global completion.
func (t *SharedTiming) MarkSearchable() {
	t.MarkSearchableAt(time.Now())
}

// MarkSearchableAt records the shared "searchable" milestone using the
// provided timestamp so elapsed-time calculations can anchor to the start
// of the successful stabilization window.
func (t *SharedTiming) MarkSearchableAt(at time.Time) {
	t.searchableOnce.Do(func() {
		t.searchableAt = at
		close(t.searchableCh)
	})
}

// WaitSearchable blocks until the shared searchable milestone is reached.
func (t *SharedTiming) WaitSearchable() {
	<-t.searchableCh
}

// MarkClientReady is mainly relevant for zero-row clients so they still take
// part in the preserved barrier/timing structure.
func (t *SharedTiming) MarkClientReady() {
	t.mu.Lock()
	t.ready++
	reachedAll := t.ready == t.expected
	t.mu.Unlock()
	if reachedAll {
		t.MarkSearchable()
	}
}

// RankZeroGate blocks non-zero ranks until rank 0 completes the gated action.
type RankZeroGate struct {
	mu     sync.Mutex
	cond   *sync.Cond
	opened bool
}

// NewRankZeroGate preserves the rank-0-first workflow marker behavior.
func NewRankZeroGate() *RankZeroGate {
	g := &RankZeroGate{}
	g.cond = sync.NewCond(&g.mu)
	return g
}

// Wait blocks non-zero ranks until rank 0 opens the gate.
func (g *RankZeroGate) Wait(globalRank int) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if globalRank == 0 {
		return
	}

	for !g.opened {
		g.cond.Wait()
	}
}

// Open releases all non-zero ranks once the gated action is done.
func (g *RankZeroGate) Open() {
	g.mu.Lock()
	g.opened = true
	g.cond.Broadcast()
	g.mu.Unlock()
}

// WriteCoordinator ensures rank 0 emits the first CSV record and serializes
// CSV appends after that so no two clients append concurrently.
type WriteCoordinator struct {
	mu               sync.Mutex
	cond             *sync.Cond
	writerActive     bool
	rankZeroFinished bool
}

// NewWriteCoordinator preserves deterministic CSV append ordering.
func NewWriteCoordinator() *WriteCoordinator {
	c := &WriteCoordinator{}
	c.cond = sync.NewCond(&c.mu)
	return c
}

// Lock serializes CSV appends and keeps rank 0 responsible for the first row.
func (c *WriteCoordinator) Lock(globalClientRank int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for c.writerActive || (!c.rankZeroFinished && globalClientRank != 0) {
		c.cond.Wait()
	}

	c.writerActive = true
}

// Unlock releases CSV append ownership and advances the rank-0-first rule.
func (c *WriteCoordinator) Unlock(globalClientRank int) {
	c.mu.Lock()
	if globalClientRank == 0 && !c.rankZeroFinished {
		c.rankZeroFinished = true
	}
	c.writerActive = false
	c.cond.Broadcast()
	c.mu.Unlock()
}

// SkipRankZero is used when rank 0 has no local rows and should not hold up
// other clients trying to write CSV records.
func (c *WriteCoordinator) SkipRankZero() {
	c.mu.Lock()
	if !c.rankZeroFinished {
		c.rankZeroFinished = true
		c.cond.Broadcast()
	}
	c.mu.Unlock()
}

// LockedCSVWriter serializes appends and ensures rank 0 emits the header first.
type LockedCSVWriter struct {
	mu            sync.Mutex
	headerWritten bool
}

// Append writes one timing CSV record and emits the header once from rank 0.
func (w *LockedCSVWriter) Append(
	resultPath string,
	taskName string,
	globalClientRank int,
	header []string,
	record []string,
) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	filename := fmt.Sprintf("%s/%s_times.csv", resultPath, strings.ToLower(strings.TrimSpace(taskName)))

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if globalClientRank == 0 && !w.headerWritten {
		if err := writer.Write(header); err != nil {
			return err
		}
		w.headerWritten = true
	}
	if err := writer.Write(record); err != nil {
		return err
	}
	writer.Flush()
	return writer.Error()
}

// splitRange splits [0, n) into `parts` contiguous chunks.
// This logic is reused unchanged because data partitioning is backend-agnostic.
func splitRange(n, parts, idx int) (start, end int) {
	if parts <= 0 {
		return 0, 0
	}
	if idx < 0 || idx >= parts {
		return 0, 0
	}
	base := n / parts
	rem := n % parts

	if idx < rem {
		start = idx * (base + 1)
		end = start + (base + 1)
	} else {
		start = rem*(base+1) + (idx-rem)*base
		end = start + base
	}
	return start, end
}

// workerAndClientForGlobalRank packs logical clients across workers using the
// configured clients-per-worker value as worker capacity.
func workerAndClientForGlobalRank(globalRank, nWorkers, clientsPerWorker int) (int, int, error) {
	if clientsPerWorker <= 0 {
		return 0, 0, fmt.Errorf("clientsPerWorker must be positive")
	}

	workerRank := globalRank / clientsPerWorker
	clientID := globalRank % clientsPerWorker
	if workerRank >= nWorkers {
		return 0, 0, fmt.Errorf(
			"global rank %d maps to worker %d, but only %d workers are available",
			globalRank,
			workerRank,
			nWorkers,
		)
	}
	return workerRank, clientID, nil
}

// clientRange preserves the historical two-stage split unless an explicit
// total query client count asks for one global split across only active clients.
func clientRange(totalRows, workerRank, clientID, nWorkers, clientsPerWorker, totalClients int, explicitTotalClients bool) (int, int) {
	if explicitTotalClients {
		return splitRange(totalRows, totalClients, workerRank*clientsPerWorker+clientID)
	}

	workerStart, workerEnd := splitRange(totalRows, nWorkers, workerRank)
	workerLen := workerEnd - workerStart
	clientStartOff, clientEndOff := splitRange(workerLen, clientsPerWorker, clientID)
	return workerStart + clientStartOff, workerStart + clientEndOff
}

// envEnabled implements the repo's permissive boolean env parsing.
func envEnabled(name string) bool {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return false
	}

	switch strings.ToLower(value) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// getEnvIntDefault returns the first positive integer found in the provided
// env vars, or the supplied default when none are set.
func getEnvIntDefault(defaultValue int, names ...string) int {
	for _, name := range names {
		value := strings.TrimSpace(os.Getenv(name))
		if value == "" {
			continue
		}

		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			log.Fatalf("invalid %s=%q", name, value)
		}
		return parsed
	}

	return defaultValue
}

// getEnvDurationSecondsDefault returns the first positive float value found in
// the provided env vars, interpreted as seconds.
func getEnvDurationSecondsDefault(defaultValue time.Duration, names ...string) time.Duration {
	for _, name := range names {
		value := strings.TrimSpace(os.Getenv(name))
		if value == "" {
			continue
		}

		parsed, err := strconv.ParseFloat(value, 64)
		if err != nil || parsed <= 0 {
			log.Fatalf("invalid %s=%q", name, value)
		}
		return time.Duration(parsed * float64(time.Second))
	}

	return defaultValue
}

// getOptionalEnvInt reads an optional integer env var without inventing a
// fallback when the user did not set one.
func getOptionalEnvInt(name string) (int, bool, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return 0, false, nil
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, true, err
	}
	return parsed, true, nil
}

// parseBatchSizes preserves the shared env parsing shape even though this
// client currently accepts exactly one batch size per run.
func parseBatchSizes(raw string) ([]int, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, fmt.Errorf("empty batch size")
	}

	if single, err := strconv.Atoi(trimmed); err == nil {
		if single <= 0 {
			return nil, fmt.Errorf("invalid batch size %q", raw)
		}
		return []int{single}, nil
	}

	cleaned := strings.Trim(trimmed, "()[]")
	fields := strings.FieldsFunc(cleaned, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\n'
	})
	if len(fields) == 0 {
		return nil, fmt.Errorf("invalid batch size list %q", raw)
	}

	sizes := make([]int, 0, len(fields))
	for _, field := range fields {
		value, err := strconv.Atoi(field)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("invalid batch size entry %q in %q", field, raw)
		}
		sizes = append(sizes, value)
	}

	return sizes, nil
}

// clientWorker keeps the Milvus batch-client orchestration shape intact:
// partitioning, barriers, timing windows, CSV output, and artifact emission.
// The transport and completion logic inside it are now Weaviate-specific.
func clientWorker(
	wg *sync.WaitGroup,
	workerRank int,
	clientID int,
	clientsPerWorker int,
	totalClients int,
	explicitTotalClients bool,
	totalRows int,
	input *InputData,
	sweeps []SweepConfig,
	barriers []*Barrier,
	sharedTimings []*SharedTiming,
	startGates []*RankZeroGate,
	writeCoordinators []*WriteCoordinator,
	resultWriters []*LockedCSVWriter,
) {
	ctx, cancel := context.WithCancel(context.Background())
	defer wg.Done()
	defer cancel()

	globalClientRank := workerRank*clientsPerWorker + clientID
	debugEnabled := envEnabled("DEBUG")
	activeTask := os.Getenv("ACTIVE_TASK")
	task := os.Getenv("TASK")
	mode := strings.ToUpper(strings.TrimSpace(activeTask))
	opMode := mode
	if opMode == "INDEX" {
		opMode = "INSERT"
	}

	nWorkersStr := os.Getenv("N_WORKERS")
	nWorkers, err := strconv.Atoi(nWorkersStr)
	if err != nil || nWorkers <= 0 {
		log.Fatalf("invalid N_WORKERS=%q", nWorkersStr)
	}

	startIdx, endIdx := clientRange(totalRows, workerRank, clientID, nWorkers, clientsPerWorker, totalClients, explicitTotalClients)
	localRows := endIdx - startIdx

	var local [][]float32
	if localRows > 0 && !input.Streaming {
		local = input.Matrix[startIdx:endIdx]
	}

	var streamFile *os.File
	if input.Streaming && localRows > 0 {
		streamFile, err = os.Open(input.Path)
		if err != nil {
			log.Fatalf("failed to open npy file for streaming: %v", err)
		}
		defer streamFile.Close()
	}

	balanceEnv := fmt.Sprintf("%s_BALANCE_STRATEGY", opMode)
	balanceStrategy := strings.ToUpper(strings.TrimSpace(os.Getenv(balanceEnv)))
	if balanceStrategy == "" {
		log.Fatalf("invalid %s=%q", balanceEnv, balanceStrategy)
	}

	var node *NodeInfo
	allNodes, err := getAllNodes("./ip_registry.txt")
	if err != nil {
		log.Fatalf("failed to read weaviate registry: %v", err)
	}

	switch balanceStrategy {
	case "NONE", "NO_BALANCE":
		node, err = getNodeByRank("./ip_registry.txt", 0)
	case "WORKER", "WORKER_BALANCE":
		node, err = getNodeByRank("./ip_registry.txt", workerRank)
	default:
		log.Fatalf("unknown %s=%q", balanceEnv, balanceStrategy)
	}

	client, err := weaviate.NewClient(weaviate.Config{
		Host:   fmt.Sprintf("%s:%d", node.IP, node.HTTPPort),
		Scheme: "http",
		GrpcConfig: &wgrpc.Config{
			Host:    fmt.Sprintf("%s:%d", node.IP, node.GRPCPort),
			Secured: false,
		},
	})

	if err != nil {
		log.Fatalf("failed to create Weaviate client: %v", err)
	}

	collectionName := os.Getenv("COLLECTION_NAME")
	localLastID := int64(endIdx - 1)
	lastID := int64(totalRows - 1)
	topK := getEnvIntDefault(10, "QUERY_TOPK", "QUERY_TOP_K", "TOP_K")
	rpcTimeout := time.Duration(getEnvIntDefault(1800, "RPC_TIMEOUT_SEC", "QUERY_RPC_SEC", "INSERT_RPC_SEC")) * time.Second
	pollInterval := getEnvDurationSecondsDefault(100*time.Millisecond, "POLL_INTERVAL")
	if rpcTimeout <= 0 {
		rpcTimeout = 30 * time.Minute
	}

	for sweepIdx, sweep := range sweeps {
		barrier := barriers[sweepIdx]
		sharedTiming := sharedTimings[sweepIdx]
		startGate := startGates[sweepIdx]
		writeCoordinator := writeCoordinators[sweepIdx]
		resultWriter := resultWriters[sweepIdx]

		fmt.Printf(
			"WeaviateWorker=%d client=%d global_client=%d assigned [%d,%d) rows=%d batch=%d target=%s:%d grpc=%d\n",
			workerRank, clientID, globalClientRank, startIdx, endIdx, localRows, sweep.BatchSize, node.IP, node.HTTPPort, node.GRPCPort,
		)

		var (
			totalDurations   []float64
			convertDurations []float64
			uploadDurations  []float64
			queryResultIDs   []int64
			queryResultWidth = -1
			queryResultRows  int
		)

		if localRows == 0 {
			if mode == "INSERT" && globalClientRank == 0 {
				writeCoordinator.SkipRankZero()
			}
			sharedTiming.MarkClientReady()
			barrier.Wait()
			barrier.Wait()
			barrier.Wait()
			sharedTiming.WaitSearchable()
			barrier.Wait()
			barrier.Wait()
			continue
		}

		barrier.Wait()

		if len(sweeps) == 1 && activeTask == task {
			startGate.Wait(globalClientRank)
			if globalClientRank == 0 {
				file, err := os.Create(runtimeStatePath("workflow_start.txt"))
				if err != nil {
					log.Fatalf("failed to create file: %v", err)
				}
				file.Close()
				startGate.Open()
			}
		}

		barrier.Wait()
		sharedTiming.MarkLoopStart()
		startLoop := time.Now()
		for i := 0; i < localRows; i += sweep.BatchSize {
			end := i + sweep.BatchSize
			if end > localRows {
				end = localRows
			}

			startTotal := time.Now()
			var batch [][]float32
			if input.Streaming {
				batch, err = readRowsFromNpy(streamFile, input.Meta, startIdx+i, end-i)
				if err != nil {
					log.Fatalf("failed to read streamed batch worker=%d client=%d absRowStart=%d sweep=%s: %v", workerRank, clientID, startIdx+i, sweep.Label, err)
				}
			} else {
				batch = local[i:end]
			}

			opCtx, opCancel := context.WithTimeout(ctx, rpcTimeout)
			var startUpload time.Time

			switch opMode {
			case "INSERT":
				objects := buildWeaviateBatchObjects(collectionName, startIdx+i, batch)
				startUpload = time.Now()
				_, err = client.Batch().ObjectsBatcher().WithObjects(objects...).Do(opCtx)
			case "QUERY":
				type queryBatchResult struct {
					ids []int64
					err error
				}

				// The Weaviate gRPC client does not expose a result shape for
				// independent batched query vectors here, so QUERY_BATCH_SIZE is
				// used as a client-side batch/concurrency limit: one search per
				// vector, launched within the current batch window.
				startUpload = time.Now()
				results := make([]queryBatchResult, len(batch))
				var queryWG sync.WaitGroup
				for j, vec := range batch {
					queryWG.Add(1)
					go func(idx int, vec []float32) {
						defer queryWG.Done()
						_, ids, _, qerr := runOneQuery(opCtx, client, collectionName, vec, topK, rpcTimeout)
						results[idx] = queryBatchResult{ids: ids, err: qerr}
					}(j, vec)
				}
				queryWG.Wait()

				for _, result := range results {
					if result.err != nil {
						err = result.err
						break
					}
					if queryResultWidth == -1 {
						queryResultWidth = len(result.ids)
					} else if len(result.ids) != queryResultWidth {
						err = fmt.Errorf("inconsistent query result width got=%d expected=%d", len(result.ids), queryResultWidth)
						break
					}
					queryResultIDs = append(queryResultIDs, result.ids...)
					queryResultRows++
				}
			default:
				log.Fatalf("unknown ACTIVE_TASK=%s", activeTask)
			}

			opCancel()
			if err != nil {
				log.Fatalf("op failed worker=%d client=%d absRowStart=%d sweep=%s: %v", workerRank, clientID, startIdx+i, sweep.Label, err)
			}

			endUpload := time.Now()
			convertDurations = append(convertDurations, startUpload.Sub(startTotal).Seconds())
			uploadDurations = append(uploadDurations, endUpload.Sub(startUpload).Seconds())
			totalDurations = append(totalDurations, endUpload.Sub(startTotal).Seconds())
		}
		endLoop := time.Now()

		if debugEnabled {
			log.Printf(
				"worker finished local %s loop worker=%d client=%d global_client=%d rows=%d abs_rows=[%d,%d) loop_seconds=%.3f target=%s:%d",
				opMode,
				workerRank,
				clientID,
				globalClientRank,
				localRows,
				startIdx,
				endIdx,
				endLoop.Sub(startLoop).Seconds(),
				node.IP,
				node.HTTPPort,
			)
		}

		barrier.Wait()

		if globalClientRank == 0 {
			sharedTiming.MarkLoopDoneAt(time.Now())
			if opMode == "INSERT" {
				lastInsertCompleteAt := endLoop
				// waitCtx, waitCancel := context.WithTimeout(ctx, *time.Minute)
				searchableAt, err := waitForQuiescence(ctx, client, allNodes, collectionName, mode == "INDEX", pollInterval)
				// waitCancel()
				if err != nil {
					log.Printf("wait for searchable collection failed: %v", err)
				} else {
					sharedTiming.MarkSearchableAt(searchableAt)
					if mode == "INDEX" {
						indexingSeconds := searchableAt.Sub(lastInsertCompleteAt).Seconds()
						if err := writeTextFile(fmt.Sprintf("%s/index_time.txt", sweep.ResultPath), fmt.Sprintf("%.6f\n", indexingSeconds)); err != nil {
							log.Printf("failed to write index_time.txt: %v", err)
						}
					}
				}
			}
			sharedTiming.MarkSearchable()
		}

		sharedTiming.WaitSearchable()

		if globalClientRank == 0 && opMode == "INSERT" {
			statusCtx, statusCancel := context.WithTimeout(ctx, 10*time.Second)
			logCollectionStatus(statusCtx, client, collectionName)
			statusCancel()
		}

		if len(sweeps) == 1 && activeTask == task && globalClientRank == 0 {
			file, err := os.Create(runtimeStatePath("workflow_end.txt"))
			if err != nil {
				log.Fatalf("failed to create file: %v", err)
			}
			file.Close()
		}
		barrier.Wait()

		localExists := false
		if opMode == "INSERT" && localRows > 0 {
			localExists, err = queryDocIDExists(ctx, client, collectionName, localLastID, rpcTimeout)
			if err != nil {
				log.Printf("Local sanity check failed worker=%d client=%d localLastID=%d: %v",
					workerRank, clientID, localLastID, err)
			}
		}

		globalExists := false
		if opMode == "INSERT" {
			globalExists, err = queryDocIDExists(ctx, client, collectionName, lastID, rpcTimeout)
			if err != nil {
				log.Printf("Global sanity check failed worker=%d client=%d lastID=%d: %v",
					workerRank, clientID, lastID, err)
			}
		}

		barrier.Wait()

		loopDuration := endLoop.Sub(startLoop).Seconds()
		waitDuration := sharedTiming.searchableAt.Sub(endLoop).Seconds()
		clientTotalToSearchable := sharedTiming.searchableAt.Sub(startLoop).Seconds()
		sharedWindow := sharedTiming.searchableAt.Sub(sharedTiming.loopStart).Seconds()
		if mode == "INDEX" {
			waitDuration = 0
			clientTotalToSearchable = loopDuration
			sharedWindow = sharedTiming.loopDoneAt.Sub(sharedTiming.loopStart).Seconds()
		}

		writeCoordinator.Lock(globalClientRank)
		resultTaskName := mode
		if opMode == "INSERT" {
			resultTaskName = "INSERT"
		}
		err = resultWriter.Append(
			sweep.ResultPath,
			resultTaskName,
			globalClientRank,
			[]string{
				"worker", "client", "global_client",
				"start_idx", "end_idx",
				"local_sanity_check", "global_sanity_check",
				"loop_duration", "wait_after_loop", "client_total_to_searchable",
				"shared_loop_start_to_searchable",
			},
			[]string{
				strconv.Itoa(workerRank),
				strconv.Itoa(clientID),
				strconv.Itoa(globalClientRank),
				strconv.Itoa(startIdx),
				strconv.Itoa(endIdx),
				strconv.FormatBool(localExists),
				strconv.FormatBool(globalExists),
				strconv.FormatFloat(loopDuration, 'g', -1, 64),
				strconv.FormatFloat(waitDuration, 'g', -1, 64),
				strconv.FormatFloat(clientTotalToSearchable, 'g', -1, 64),
				strconv.FormatFloat(sharedWindow, 'g', -1, 64),
			},
		)
		writeCoordinator.Unlock(globalClientRank)
		if err != nil {
			log.Fatal(err)
		}

		w1, _ := gonpy.NewFileWriter(fmt.Sprintf(sweep.ResultPath+"/batch_construction_times_w%d_c%d.npy", workerRank, clientID))
		_ = w1.WriteFloat64(convertDurations)

		w2, _ := gonpy.NewFileWriter(fmt.Sprintf(sweep.ResultPath+"/upload_times_w%d_c%d.npy", workerRank, clientID))
		_ = w2.WriteFloat64(uploadDurations)

		w3, _ := gonpy.NewFileWriter(fmt.Sprintf(sweep.ResultPath+"/op_times_w%d_c%d.npy", workerRank, clientID))
		_ = w3.WriteFloat64(totalDurations)

		if mode == "QUERY" {
			if queryResultWidth < 0 {
				queryResultWidth = 0
			}
			queryIDsPath := fmt.Sprintf(sweep.ResultPath+"/query_result_ids_w%d_c%d.npy", workerRank, clientID)
			if err := writeInt64Npy(queryIDsPath, queryResultIDs, []int{queryResultRows, queryResultWidth}); err != nil {
				log.Fatalf("failed to write query result ids worker=%d client=%d sweep=%s: %v", workerRank, clientID, sweep.Label, err)
			}
		}
	}
}

func main() {
	nWorkersStr := os.Getenv("N_WORKERS")
	nWorkers, err := strconv.Atoi(nWorkersStr)
	if err != nil || nWorkers <= 0 {
		log.Fatalf("invalid N_WORKERS=%q", nWorkersStr)
	}

	activeTask := os.Getenv("ACTIVE_TASK")
	task := os.Getenv("TASK")
	if activeTask == "" && task == "" {
		log.Fatal("ACTIVE_TASK and TASK environment variables are not set")
	}
	if activeTask == "" {
		activeTask = task
	}

	mode := strings.ToUpper(strings.TrimSpace(activeTask))
	workloadMode := mode
	if workloadMode == "INDEX" {
		workloadMode = "INSERT"
	}
	if workloadMode != "INSERT" && workloadMode != "QUERY" {
		log.Fatalf("unsupported ACTIVE_TASK=%q; batch_client only supports INSERT, INDEX, or QUERY", activeTask)
	}
	fmt.Printf("Active task: %s (TASK=%s mode=%s)\n", activeTask, task, mode)

	clientsEnv := fmt.Sprintf("%s_CLIENTS_PER_WORKER", workloadMode)
	clientsStr := os.Getenv(clientsEnv)
	clientsPerWorker, err := strconv.Atoi(clientsStr)
	if err != nil || clientsPerWorker <= 0 {
		log.Fatalf("invalid %s=%q", clientsEnv, clientsStr)
	}

	totalClients := nWorkers * clientsPerWorker
	explicitTotalClients := false
	totalClientsEnv := fmt.Sprintf("TOTAL_%s_CLIENTS", workloadMode)
	configuredTotalClients, isSet, err := getOptionalEnvInt(totalClientsEnv)
	if err != nil || (isSet && configuredTotalClients <= 0) {
		log.Fatalf("invalid %s=%q", totalClientsEnv, os.Getenv(totalClientsEnv))
	}
	if isSet {
		if configuredTotalClients > totalClients {
			log.Fatalf(
				"%s=%d exceeds capacity N_WORKERS * %s = %d",
				totalClientsEnv,
				configuredTotalClients,
				clientsEnv,
				totalClients,
			)
		}
		totalClients = configuredTotalClients
		explicitTotalClients = true
	}

	corpusEnv := fmt.Sprintf("%s_CORPUS_SIZE", workloadMode)
	corpusSizeStr := strings.TrimSpace(os.Getenv(corpusEnv))
	corpusSize := 0
	if corpusSizeStr != "" {
		corpusSize, err = strconv.Atoi(corpusSizeStr)
		if err != nil || corpusSize <= 0 {
			log.Fatalf("invalid %s=%q", corpusEnv, corpusSizeStr)
		}
	}

	dataPathEnv := fmt.Sprintf("%s_DATA_FILEPATH", workloadMode)
	dataPath := os.Getenv(dataPathEnv)

	batchEnv := fmt.Sprintf("%s_BATCH_SIZE", workloadMode)
	batchSizeStr := os.Getenv(batchEnv)
	batchSizes, err := parseBatchSizes(batchSizeStr)
	if err != nil {
		log.Fatalf("invalid %s=%q: %v", batchEnv, batchSizeStr, err)
	}
	if len(batchSizes) != 1 {
		log.Fatalf("%s must contain exactly one batch size for now", batchEnv)
	}
	batchSize := batchSizes[0]
	resultPath := "."

	sweeps := []SweepConfig{{
		BatchSize:  batchSize,
		ResultPath: resultPath,
		Label:      fmt.Sprintf("batch_%d", batchSize),
	}}

	fmt.Printf(
		"CORPUS_SIZE=%d nWorkers=%d clientsPerWorker=%d totalClients=%d explicitTotalClients=%t DATA_FILEPATH=%s batch_size=%d\n",
		corpusSize, nWorkers, clientsPerWorker, totalClients, explicitTotalClients, dataPath, batchSize,
	)

	vectorDim, hasVectorDim, err := getOptionalEnvInt("VECTOR_DIM")
	if err != nil {
		log.Fatalf("invalid VECTOR_DIM=%q", os.Getenv("VECTOR_DIM"))
	}

	activeTaskStreamingEnv := fmt.Sprintf("%s_STREAMING", strings.ToUpper(strings.TrimSpace(activeTask)))
	streamingReads := envEnabled(activeTaskStreamingEnv)
	if !streamingReads && workloadMode == "INSERT" {
		// INDEX reuses the insert workload path, so inherit INSERT_STREAMING when
		// the caller did not define a separate INDEX_STREAMING knob.
		streamingReads = envEnabled("INSERT_STREAMING")
	}
	input := &InputData{
		Streaming: streamingReads,
		Path:      dataPath,
	}
	if streamingReads {
		meta, err := parseNpyMetadata(dataPath)
		if err != nil {
			log.Fatalf("failed to parse npy metadata for %s: %v", dataPath, err)
		}
		if hasVectorDim && vectorDim != meta.Cols {
			log.Fatalf("VECTOR_DIM mismatch: env=%d npy=%d", vectorDim, meta.Cols)
		}
		if corpusSize == 0 {
			corpusSize = meta.Rows
		}
		if corpusSize > meta.Rows {
			log.Fatalf("corpus size %d exceeds npy row count %d", corpusSize, meta.Rows)
		}
		meta.Rows = corpusSize
		input.Meta = meta
		fmt.Printf("Input mode: streaming path=%s rows=%d cols=%d\n", dataPath, meta.Rows, meta.Cols)
	} else {
		r, err := gonpy.NewFileReader(dataPath)
		if err != nil {
			log.Fatalf("failed to open npy file %s: %v", dataPath, err)
		}
		shape := r.Shape
		if len(shape) != 2 {
			log.Fatalf("expected 2D npy input, got shape=%v", shape)
		}
		if hasVectorDim && vectorDim != shape[1] {
			log.Fatalf("VECTOR_DIM mismatch: env=%d npy=%d", vectorDim, shape[1])
		}
		if corpusSize == 0 {
			corpusSize = shape[0]
		}
		if corpusSize > shape[0] {
			log.Fatalf("corpus size %d exceeds npy row count %d", corpusSize, shape[0])
		}

		cols := shape[1]
		data, err := r.GetFloat32()
		if err != nil {
			log.Fatalf("failed to read npy data from %s: %v", dataPath, err)
		}

		matrix := make([][]float32, corpusSize)
		for i := 0; i < corpusSize; i++ {
			start := i * cols
			end := start + cols
			matrix[i] = data[start:end]
		}
		input.Matrix = matrix
		fmt.Printf("Input mode: eager path=%s rows=%d cols=%d\n", dataPath, corpusSize, cols)
	}

	var wg sync.WaitGroup
	barriers := make([]*Barrier, len(sweeps))
	sharedTimings := make([]*SharedTiming, len(sweeps))
	startGates := make([]*RankZeroGate, len(sweeps))
	writeCoordinators := make([]*WriteCoordinator, len(sweeps))
	resultWriters := make([]*LockedCSVWriter, len(sweeps))
	for i := range sweeps {
		barriers[i] = NewBarrier(totalClients)
		sharedTimings[i] = NewSharedTiming(totalClients)
		startGates[i] = NewRankZeroGate()
		writeCoordinators[i] = NewWriteCoordinator()
		resultWriters[i] = &LockedCSVWriter{}
	}

	for globalRank := 0; globalRank < totalClients; globalRank++ {
		workerRank, clientID, err := workerAndClientForGlobalRank(globalRank, nWorkers, clientsPerWorker)
		if err != nil {
			log.Fatal(err)
		}
		wg.Add(1)
		go clientWorker(
			&wg,
			workerRank,
			clientID,
			clientsPerWorker,
			totalClients,
			explicitTotalClients,
			corpusSize,
			input,
			sweeps,
			barriers,
			sharedTimings,
			startGates,
			writeCoordinators,
			resultWriters,
		)
	}

	wg.Wait()
	fmt.Printf("%s complete\n", activeTask)
}
