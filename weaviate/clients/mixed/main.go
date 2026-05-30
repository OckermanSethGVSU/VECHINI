package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/kshedden/gonpy"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	wgraphql "github.com/weaviate/weaviate-go-client/v5/weaviate/graphql"
	wgrpc "github.com/weaviate/weaviate-go-client/v5/weaviate/grpc"
	"github.com/weaviate/weaviate/entities/models"
)

type role string

const (
	roleInsert role = "insert"
	roleQuery  role = "query"
)

type runMode string

const (
	modeMax  runMode = "max"
	modeRate runMode = "rate"
)

type config struct {
	// Connection, schema, and input/output paths.
	collectionName   string
	insertStartID    int64
	outputDir        string
	insertVectors    string
	queryVectors     string
	insertCorpusSize int
	queryCorpusSize  int
	insertClients    int
	queryClients     int
	topK             int
	ef               int
	mode             runMode
	insertMode       runMode
	queryMode        runMode
	insertOpsPerSec  float64
	queryOpsPerSec   float64
	rpcTimeout       time.Duration

	// Batch behavior is configured separately for inserts and queries.
	insertBatch batchConfig
	queryBatch  batchConfig
}

type batchConfig struct {
	fixed int
	min   int
	max   int
}

type batchPicker struct {
	cfg batchConfig
	rng *rand.Rand
}

type matrixData struct {
	// data stores a dense row-major matrix loaded from a 2D .npy file.
	rows int
	dim  int
	data []float32
}

type clientAssignment struct {
	// Each client owns a contiguous slice of the input matrix.
	clientID int
	startRow int
	endRow   int
}

type insertLogRecord struct {
	ClientRole      role   `json:"client_role"`
	ClientID        int    `json:"client_id"`
	OpIndex         int    `json:"op_index"`
	IssuedAtNS      int64  `json:"issued_at_ns"`
	CompletedAtNS   int64  `json:"completed_at_ns"`
	DurationNS      int64  `json:"duration_ns"`
	BatchStartRow   int    `json:"batch_start_row"`
	BatchEndRow     int    `json:"batch_end_row"`
	InsertedIDStart int64  `json:"inserted_id_start"`
	InsertedIDEnd   int64  `json:"inserted_id_end_exclusive"`
	Status          string `json:"status"`
	Error           string `json:"error,omitempty"`
}

type queryLogRecord struct {
	ClientRole      role        `json:"client_role"`
	ClientID        int         `json:"client_id"`
	OpIndex         int         `json:"op_index"`
	IssuedAtNS      int64       `json:"issued_at_ns"`
	CompletedAtNS   int64       `json:"completed_at_ns"`
	DurationNS      int64       `json:"duration_ns"`
	QueryStartRow   int         `json:"query_start_row"`
	QueryEndRow     int         `json:"query_end_row"`
	QueryRowIndices []int       `json:"query_row_indices"`
	ResultIDs       [][]int64   `json:"result_ids"`
	ResultScores    [][]float32 `json:"result_scores"`
	Status          string      `json:"status"`
	Error           string      `json:"error,omitempty"`
}

type queryResult struct {
	IDs    []int64
	Scores []float32
}

type nodeTarget struct {
	Rank     int
	Node     string
	IP       string
	HTTPPort int
	GRPCPort int
}

type backend interface {
	// Load lets each backend do one-time setup before the client loop starts.
	Load(ctx context.Context) error
	Insert(ctx context.Context, ids []int64, vectors [][]float32) error
	Search(ctx context.Context, vectors [][]float32, topK int) ([]queryResult, error)
	Close(ctx context.Context) error
}

type backendFactory func(clientID int, r role) (backend, error)

type weaviateBackend struct {
	client         *weaviate.Client
	collectionName string
}

const idPropName = "doc_id"

func main() {
	cfg, err := parseFlags()
	if err != nil {
		log.Fatal(err)
	}

	if err := os.MkdirAll(cfg.outputDir, 0o755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Insert and query workloads can come from different vector corpora.
	insertData, err := loadMatrix(cfg.insertVectors)
	if err != nil {
		log.Fatalf("load insert vectors: %v", err)
	}
	if cfg.insertCorpusSize > insertData.rows {
		log.Fatalf("insert corpus size %d exceeds available rows %d", cfg.insertCorpusSize, insertData.rows)
	}

	// chop down if needed
	if cfg.insertCorpusSize > 0 && cfg.insertCorpusSize < insertData.rows {
		insertData.rows = cfg.insertCorpusSize
		insertData.data = insertData.data[:cfg.insertCorpusSize*insertData.dim]
	}

	queryData, err := loadMatrix(cfg.queryVectors)
	if err != nil {
		log.Fatalf("load query vectors: %v", err)
	}
	if cfg.queryCorpusSize > queryData.rows {
		log.Fatalf("query corpus size %d exceeds available rows %d", cfg.queryCorpusSize, queryData.rows)
	}
	// chop down if needed
	if cfg.queryCorpusSize > 0 && cfg.queryCorpusSize < queryData.rows {
		queryData.rows = cfg.queryCorpusSize
		queryData.data = queryData.data[:cfg.queryCorpusSize*queryData.dim]
	}
	if insertData.dim != queryData.dim {
		log.Fatalf("dimension mismatch: insert dim=%d query dim=%d", insertData.dim, queryData.dim)
	}

	// newBackend constructs the real Weaviate client backend.
	ctx := context.Background()
	newBackend, err := newBackendFactory(ctx, cfg)
	if err != nil {
		log.Fatalf("create backend: %v", err)
	}

	if err := runWorkload(ctx, cfg, insertData, queryData, newBackend); err != nil {
		log.Fatal(err)
	}
}

func parseFlags() (config, error) {
	var cfg config
	var mode string

	// Flags default to environment variables so the runner can be driven from shell scripts.
	flag.StringVar(&cfg.collectionName, "collection", getenvDefault("COLLECTION_NAME", "DEFAULT_COLLECTION"), "Weaviate collection name")
	flag.StringVar(&mode, "mode", string(modeMax), "Execution mode: max or rate")
	flag.StringVar(&cfg.outputDir, "output-dir", getenvDefault("MIXED_RESULT_PATH", getenvDefault("RESULT_PATH", "mixed_logs")), "Directory for per-client JSONL logs")
	flag.DurationVar(&cfg.rpcTimeout, "rpc-timeout", getenvDurationDefault("RPC_TIMEOUT", 10*time.Minute), "Per-operation timeout")

	flag.StringVar((*string)(&cfg.insertMode), "insert-mode", getenvDefault("MIXED_INSERT_MODE", ""), "Per-role insert mode override: max or rate")
	flag.StringVar(&cfg.insertVectors, "insert-vectors", getenvDefault("MIXED_INSERT_DATA_FILEPATH", ""), "Path to insert vectors .npy")
	flag.Int64Var(&cfg.insertStartID, "insert-start-id", getenvInt64Default("MIXED_INSERT_START_ID", 0), "Starting ID offset for inserted vectors")
	flag.IntVar(&cfg.insertCorpusSize, "insert-corpus-size", getenvIntDefault("MIXED_INSERT_CORPUS_SIZE", 0), "Rows to read from the insert matrix; 0 means all rows")
	flag.IntVar(&cfg.insertClients, "insert-clients", getenvIntDefault("MIXED_INSERT_CLIENTS", 1), "Number of dedicated insert clients")
	flag.IntVar(&cfg.insertBatch.fixed, "insert-batch-size", getenvIntDefault("MIXED_INSERT_BATCH_SIZE", 1), "Fixed insert batch size")

	flag.StringVar((*string)(&cfg.queryMode), "query-mode", getenvDefault("MIXED_QUERY_MODE", ""), "Per-role query mode override: max or rate")
	flag.StringVar(&cfg.queryVectors, "query-vectors", getenvDefault("MIXED_QUERY_DATA_FILEPATH", ""), "Path to query vectors .npy")
	flag.IntVar(&cfg.queryCorpusSize, "query-corpus-size", getenvIntDefault("MIXED_QUERY_CORPUS_SIZE", 0), "Rows to read from the query matrix; 0 means all rows")
	flag.IntVar(&cfg.queryClients, "query-clients", getenvIntDefault("MIXED_QUERY_CLIENTS", 1), "Number of dedicated query clients")
	flag.IntVar(&cfg.queryBatch.fixed, "query-batch-size", getenvIntDefault("MIXED_QUERY_BATCH_SIZE", 1), "Fixed query batch size")

	flag.IntVar(&cfg.topK, "top-k", getenvIntDefault("TOP_K", 10), "Top-k results per query vector")
	flag.IntVar(&cfg.ef, "ef", getenvIntDefault("HNSW_EF_SEARCH", getenvIntDefault("EFSearch", 64)), "Search ef parameter")

	flag.Float64Var(&cfg.insertOpsPerSec, "insert-ops-per-sec", getenvFloatDefault("MIXED_INSERT_OPS_PER_SEC", 0), "Direct insert ops/sec cap across all insert clients")
	flag.Float64Var(&cfg.queryOpsPerSec, "query-ops-per-sec", getenvFloatDefault("MIXED_QUERY_OPS_PER_SEC", 0), "Direct query ops/sec cap across all query clients")
	flag.IntVar(&cfg.insertBatch.min, "insert-batch-min", getenvIntDefault("MIXED_INSERT_BATCH_MIN", getenvIntDefault("INSERT_BATCH_MIN", 0)), "Random insert batch min, inclusive")
	flag.IntVar(&cfg.insertBatch.max, "insert-batch-max", getenvIntDefault("MIXED_INSERT_BATCH_MAX", getenvIntDefault("INSERT_BATCH_MAX", 0)), "Random insert batch max, inclusive")
	flag.IntVar(&cfg.queryBatch.min, "query-batch-min", getenvIntDefault("MIXED_QUERY_BATCH_MIN", getenvIntDefault("QUERY_BATCH_MIN", 0)), "Random query batch min, inclusive")
	flag.IntVar(&cfg.queryBatch.max, "query-batch-max", getenvIntDefault("MIXED_QUERY_BATCH_MAX", getenvIntDefault("QUERY_BATCH_MAX", 0)), "Random query batch max, inclusive")
	flag.Parse()

	// The top-level mode acts as a default unless a role-specific mode overrides it.
	cfg.mode = normalizeRunMode(mode)
	cfg.insertMode = normalizeRunMode(string(cfg.insertMode))
	cfg.queryMode = normalizeRunMode(string(cfg.queryMode))
	if cfg.insertMode == "" {
		cfg.insertMode = cfg.mode
	}
	if cfg.queryMode == "" {
		cfg.queryMode = cfg.mode
	}

	if err := validateConfig(cfg); err != nil {
		return config{}, err
	}
	return cfg, nil
}

func normalizeRunMode(value string) runMode {
	return runMode(strings.ToLower(strings.TrimSpace(value)))
}

func validateConfig(cfg config) error {
	// Validation is centralized here so the worker loops can assume a consistent config shape.
	if cfg.insertVectors == "" {
		return errors.New("insert-vectors is required")
	}
	if cfg.queryVectors == "" {
		return errors.New("query-vectors is required")
	}
	if cfg.outputDir == "" {
		return errors.New("output-dir is required")
	}
	if cfg.insertClients <= 0 {
		return errors.New("insert-clients must be positive")
	}
	if cfg.queryClients <= 0 {
		return errors.New("query-clients must be positive")
	}
	if cfg.topK <= 0 {
		return errors.New("top-k must be positive")
	}
	if cfg.ef <= 0 {
		return errors.New("ef must be positive")
	}
	if cfg.insertCorpusSize < 0 {
		return errors.New("insert-corpus-size must be non-negative")
	}
	if cfg.queryCorpusSize < 0 {
		return errors.New("query-corpus-size must be non-negative")
	}
	if cfg.rpcTimeout <= 0 {
		return errors.New("rpc-timeout must be positive")
	}
	if cfg.mode != modeMax && cfg.mode != modeRate {
		return fmt.Errorf("unsupported mode %q", cfg.mode)
	}
	if err := validateRoleMode("insert", cfg.insertMode, cfg.insertOpsPerSec); err != nil {
		return err
	}
	if err := validateRoleMode("query", cfg.queryMode, cfg.queryOpsPerSec); err != nil {
		return err
	}
	if err := validateBatchConfig("insert", cfg.insertBatch); err != nil {
		return err
	}
	if err := validateBatchConfig("query", cfg.queryBatch); err != nil {
		return err
	}
	return nil
}

func validateRoleMode(roleName string, mode runMode, opsPerSec float64) error {
	switch mode {
	case modeMax:
		return nil
	case modeRate:
		if opsPerSec <= 0 {
			return fmt.Errorf("%s-ops-per-sec must be positive when %s-mode=rate", roleName, roleName)
		}
		return nil
	default:
		return fmt.Errorf("unsupported %s-mode %q", roleName, mode)
	}
}

func validateBatchConfig(name string, cfg batchConfig) error {
	if cfg.fixed <= 0 {
		return fmt.Errorf("%s fixed batch size must be positive", name)
	}
	if (cfg.min == 0) != (cfg.max == 0) {
		return fmt.Errorf("%s batch min/max must both be set or both be unset", name)
	}
	if cfg.min < 0 || cfg.max < 0 {
		return fmt.Errorf("%s batch min/max must be non-negative", name)
	}
	if cfg.min > 0 && cfg.min > cfg.max {
		return fmt.Errorf("%s batch min must be <= max", name)
	}
	return nil
}

func runWorkload(ctx context.Context, cfg config, insertData, queryData matrixData, newBackend backendFactory) error {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	var readyWG sync.WaitGroup
	errCh := make(chan error, cfg.insertClients+cfg.queryClients)
	startCh := make(chan struct{})
	totalClients := cfg.insertClients + cfg.queryClients
	readyWG.Add(totalClients)

	// Rows are split once up front so each client works on a disjoint slice of the corpus.
	insertAssignments := buildAssignments(insertData.rows, cfg.insertClients)
	queryAssignments := buildAssignments(queryData.rows, cfg.queryClients)

	for _, assignment := range insertAssignments {
		wg.Add(1)
		go func(a clientAssignment) {
			defer wg.Done()
			if err := runInsertClient(runCtx, cfg, insertData, newBackend, a, &readyWG, startCh); err != nil {
				errCh <- fmt.Errorf("insert client %d: %w", a.clientID, err)
				cancel()
			}
		}(assignment)
	}

	for _, assignment := range queryAssignments {
		wg.Add(1)
		go func(a clientAssignment) {
			defer wg.Done()
			if err := runQueryClient(runCtx, cfg, queryData, newBackend, a, &readyWG, startCh); err != nil {
				errCh <- fmt.Errorf("query client %d: %w", a.clientID, err)
				cancel()
			}
		}(assignment)
	}

	readyWG.Wait()
	close(startCh)

	wg.Wait()
	close(errCh)

	var joined error
	for err := range errCh {
		joined = errors.Join(joined, err)
	}
	return joined
}

func logWorkerDone(r role, assignment clientAssignment, ops int) {
	log.Printf("worker_done role=%s client=%d rows=[%d,%d) ops=%d", r, assignment.clientID, assignment.startRow, assignment.endRow, ops)
}

func logWorkerStartup(cfg config, r role, assignment clientAssignment) {
	target, err := pickNodeTarget(r, assignment.clientID)
	targetDesc := ""
	if err != nil {
		targetDesc = fmt.Sprintf("target-error=%v", err)
	} else {
		targetDesc = fmt.Sprintf("node_rank=%d target=%s:%d grpc=%d", target.Rank, target.IP, target.HTTPPort, target.GRPCPort)
	}

	rowCount := assignment.endRow - assignment.startRow
	switch r {
	case roleInsert:
		idStart := cfg.insertStartID + int64(assignment.startRow)
		idEndExclusive := cfg.insertStartID + int64(assignment.endRow)
		log.Printf("worker_ready role=%s client=%d rows=[%d,%d) row_count=%d ids=[%d,%d) %s", r, assignment.clientID, assignment.startRow, assignment.endRow, rowCount, idStart, idEndExclusive, targetDesc)
	case roleQuery:
		log.Printf("worker_ready role=%s client=%d rows=[%d,%d) row_count=%d ids=[n/a] %s", r, assignment.clientID, assignment.startRow, assignment.endRow, rowCount, targetDesc)
	default:
		log.Printf("worker_ready role=%s client=%d rows=[%d,%d) row_count=%d %s", r, assignment.clientID, assignment.startRow, assignment.endRow, rowCount, targetDesc)
	}
}

func runInsertClient(ctx context.Context, cfg config, data matrixData, newBackend backendFactory, assignment clientAssignment, readyWG *sync.WaitGroup, startCh <-chan struct{}) error {
	picker := newBatchPicker(cfg.insertBatch, roleInsert, assignment.clientID)
	pacer := newPacer(cfg, roleInsert)
	logPath := filepath.Join(cfg.outputDir, fmt.Sprintf("insert_client_%03d.jsonl", assignment.clientID))
	records := make([]insertLogRecord, 0, max(1, assignment.endRow-assignment.startRow))
	b, err := newBackend(assignment.clientID, roleInsert)
	if err != nil {
		readyWG.Done()
		return err
	}
	defer b.Close(ctx)
	if err := b.Load(ctx); err != nil {
		readyWG.Done()
		return err
	}
	logWorkerStartup(cfg, roleInsert, assignment)
	readyWG.Done()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-startCh:
	}

	opIndex := 0
	for row := assignment.startRow; row < assignment.endRow; {
		// Batch size can be fixed or pseudo-random, but never runs past this client's assigned rows.
		batchSize := picker.Next()
		if batchSize > assignment.endRow-row {
			batchSize = assignment.endRow - row
		}

		// Wait enforces the configured target rate; in max mode it is a no-op.
		if err := pacer.Wait(ctx, opIndex); err != nil {
			return err
		}

		batchVectors := rowsToSlices(data, row, row+batchSize)
		ids := make([]int64, batchSize)
		for i := 0; i < batchSize; i++ {
			// IDs are derived directly from row offsets so inserts are deterministic across runs.
			ids[i] = cfg.insertStartID + int64(row+i)
		}

		// Each RPC gets its own timeout, but the outer client context still governs the whole run.
		opCtx, cancel := context.WithTimeout(ctx, cfg.rpcTimeout)
		issuedAt := time.Now()
		err := b.Insert(opCtx, ids, batchVectors)
		completedAt := time.Now()
		cancel()

		record := insertLogRecord{
			ClientRole:      roleInsert,
			ClientID:        assignment.clientID,
			OpIndex:         opIndex,
			IssuedAtNS:      issuedAt.UnixNano(),
			CompletedAtNS:   completedAt.UnixNano(),
			DurationNS:      completedAt.Sub(issuedAt).Nanoseconds(),
			BatchStartRow:   row,
			BatchEndRow:     row + batchSize,
			InsertedIDStart: ids[0],
			InsertedIDEnd:   ids[len(ids)-1] + 1,
			Status:          "ok",
		}
		if err != nil {
			record.Status = "error"
			record.Error = err.Error()
		}
		records = append(records, record)
		if err != nil {
			// Persist the partial log before exiting so failed runs still leave a trace.
			return writeJSONL(logPath, records)
		}

		row += batchSize
		opIndex++
	}
	if err := writeJSONL(logPath, records); err != nil {
		return err
	}
	logWorkerDone(roleInsert, assignment, opIndex)
	return nil
}

func runQueryClient(ctx context.Context, cfg config, data matrixData, newBackend backendFactory, assignment clientAssignment, readyWG *sync.WaitGroup, startCh <-chan struct{}) error {
	picker := newBatchPicker(cfg.queryBatch, roleQuery, assignment.clientID)
	pacer := newPacer(cfg, roleQuery)
	logPath := filepath.Join(cfg.outputDir, fmt.Sprintf("query_client_%03d.jsonl", assignment.clientID))
	records := make([]queryLogRecord, 0, max(1, assignment.endRow-assignment.startRow))
	b, err := newBackend(assignment.clientID, roleQuery)
	if err != nil {
		readyWG.Done()
		return err
	}
	defer b.Close(ctx)
	if err := b.Load(ctx); err != nil {
		readyWG.Done()
		return err
	}
	logWorkerStartup(cfg, roleQuery, assignment)
	readyWG.Done()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-startCh:
	}

	opIndex := 0
	for row := assignment.startRow; row < assignment.endRow; {
		batchSize := picker.Next()
		if batchSize > assignment.endRow-row {
			batchSize = assignment.endRow - row
		}

		if err := pacer.Wait(ctx, opIndex); err != nil {
			return err
		}

		batchVectors := rowsToSlices(data, row, row+batchSize)
		queryRows := make([]int, 0, batchSize)
		for i := row; i < row+batchSize; i++ {
			// The log stores source row indices so results can be correlated back to the input matrix.
			queryRows = append(queryRows, i)
		}

		opCtx, cancel := context.WithTimeout(ctx, cfg.rpcTimeout)
		issuedAt := time.Now()
		results, err := b.Search(opCtx, batchVectors, cfg.topK)
		completedAt := time.Now()
		cancel()

		record := queryLogRecord{
			ClientRole:      roleQuery,
			ClientID:        assignment.clientID,
			OpIndex:         opIndex,
			IssuedAtNS:      issuedAt.UnixNano(),
			CompletedAtNS:   completedAt.UnixNano(),
			DurationNS:      completedAt.Sub(issuedAt).Nanoseconds(),
			QueryStartRow:   row,
			QueryEndRow:     row + batchSize,
			QueryRowIndices: queryRows,
			Status:          "ok",
		}
		if err == nil {
			// Copy result slices so the log owns its data even if the backend reuses buffers internally.
			record.ResultIDs = make([][]int64, 0, len(results))
			record.ResultScores = make([][]float32, 0, len(results))
			for _, res := range results {
				record.ResultIDs = append(record.ResultIDs, append([]int64(nil), res.IDs...))
				record.ResultScores = append(record.ResultScores, append([]float32(nil), res.Scores...))
			}
		} else {
			record.Status = "error"
			record.Error = err.Error()
		}
		records = append(records, record)
		if err != nil {
			return writeJSONL(logPath, records)
		}

		row += batchSize
		opIndex++
	}
	if err := writeJSONL(logPath, records); err != nil {
		return err
	}
	logWorkerDone(roleQuery, assignment, opIndex)
	return nil
}

type pacer struct {
	// startTime anchors opIndex=0 so later operations can target evenly spaced issue times.
	mode      runMode
	rate      float64
	startTime time.Time
}

func newPacer(cfg config, r role) *pacer {
	p := &pacer{startTime: time.Now()}
	// The pacer converts the global config into a per-client rate for one specific role.
	switch r {
	case roleInsert:
		p.mode = cfg.insertMode
		if cfg.insertMode == modeRate {
			p.rate = derivePerClientRate(cfg.insertOpsPerSec, cfg.insertClients)
		}
	case roleQuery:
		p.mode = cfg.queryMode
		if cfg.queryMode == modeRate {
			p.rate = derivePerClientRate(cfg.queryOpsPerSec, cfg.queryClients)
		}
	}
	return p
}

func (p *pacer) Wait(ctx context.Context, opIndex int) error {
	if p.mode != modeRate || p.rate <= 0 {
		return nil
	}
	// opIndex/rate gives the ideal elapsed time for this operation on a perfectly paced client.
	target := p.startTime.Add(time.Duration(float64(opIndex) / p.rate * float64(time.Second)))
	delay := time.Until(target)
	if delay <= 0 {
		// If work is already behind schedule, the client immediately issues the next operation.
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func derivePerClientRate(totalOpsPerSec float64, clients int) float64 {
	if clients <= 0 {
		return 0
	}
	return totalOpsPerSec / float64(clients)
}

func newBatchPicker(cfg batchConfig, r role, clientID int) batchPicker {
	seedBase := int64(clientID + 1)
	if r == roleQuery {
		seedBase += 10_000
	}
	// The seed is deterministic per role/client pair so randomized batches are reproducible.
	return batchPicker{
		cfg: cfg,
		rng: rand.New(rand.NewSource(seedBase)),
	}
}

func (p batchPicker) Next() int {
	if p.cfg.min > 0 && p.cfg.max > 0 {
		if p.cfg.min == p.cfg.max {
			return p.cfg.min
		}
		return p.cfg.min + p.rng.Intn(p.cfg.max-p.cfg.min+1)
	}
	return p.cfg.fixed
}

func buildAssignments(totalRows, clients int) []clientAssignment {
	assignments := make([]clientAssignment, 0, clients)
	for i := 0; i < clients; i++ {
		// splitRange distributes any remainder to the earliest clients.
		start, end := splitRange(totalRows, clients, i)
		assignments = append(assignments, clientAssignment{
			clientID: i,
			startRow: start,
			endRow:   end,
		})
	}
	return assignments
}

func splitRange(total, parts, idx int) (int, int) {
	if parts <= 0 || idx < 0 || idx >= parts {
		return 0, 0
	}
	base := total / parts
	rem := total % parts
	if idx < rem {
		start := idx * (base + 1)
		return start, start + base + 1
	}
	start := rem*(base+1) + (idx-rem)*base
	return start, start + base
}

func loadMatrix(path string) (matrixData, error) {
	// The runner expects a simple 2D float32 matrix where each row is one vector.
	reader, err := gonpy.NewFileReader(path)
	if err != nil {
		return matrixData{}, err
	}
	if len(reader.Shape) != 2 {
		return matrixData{}, fmt.Errorf("%s must be 2D, got shape %v", path, reader.Shape)
	}
	values, err := reader.GetFloat32()
	if err != nil {
		return matrixData{}, err
	}
	rows := reader.Shape[0]
	dim := reader.Shape[1]
	expected := rows * dim
	if len(values) != expected {
		return matrixData{}, fmt.Errorf("%s has %d values, expected %d", path, len(values), expected)
	}
	return matrixData{rows: rows, dim: dim, data: values}, nil
}

func getenvDefault(name, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}

func getenvIntDefault(name string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func getenvInt64Default(name string, fallback int64) int64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func getenvFloatDefault(name string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func getenvDurationDefault(name string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func getenvBoolDefault(name string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	switch strings.ToLower(value) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func rowsToSlices(data matrixData, start, end int) [][]float32 {
	rows := make([][]float32, 0, end-start)
	for i := start; i < end; i++ {
		offset := i * data.dim
		// These slices alias the original matrix storage to avoid copying vector data on every op.
		rows = append(rows, data.data[offset:offset+data.dim])
	}
	return rows
}

func newBackendFactory(ctx context.Context, cfg config) (backendFactory, error) {
	return func(clientID int, r role) (backend, error) {
		// Each logical client gets its own Weaviate connection and may be routed to a different node.
		target, err := pickNodeTarget(r, clientID)
		if err != nil {
			return nil, err
		}
		client, err := weaviate.NewClient(weaviate.Config{
			Host:   fmt.Sprintf("%s:%d", target.IP, target.HTTPPort),
			Scheme: "http",
			GrpcConfig: &wgrpc.Config{
				Host:    fmt.Sprintf("%s:%d", target.IP, target.GRPCPort),
				Secured: false,
			},
		})
		if err != nil {
			return nil, err
		}
		return &weaviateBackend{
			client:         client,
			collectionName: cfg.collectionName,
		}, nil
	}, nil
}

func (w *weaviateBackend) Load(context.Context) error {
	return nil
}

func (w *weaviateBackend) Insert(ctx context.Context, ids []int64, vectors [][]float32) error {
	objects := make([]*models.Object, 0, len(vectors))
	for i, vec := range vectors {
		objectVector := make([]float32, len(vec))
		copy(objectVector, vec)
		objects = append(objects, &models.Object{
			Class: w.collectionName,
			Properties: map[string]any{
				idPropName: ids[i],
			},
			Vector: objectVector,
		})
	}
	_, err := w.client.Batch().ObjectsBatcher().WithObjects(objects...).Do(ctx)
	return err
}

func (w *weaviateBackend) Search(ctx context.Context, vectors [][]float32, topK int) ([]queryResult, error) {
	results := make([]queryResult, len(vectors))
	errCh := make(chan error, len(vectors))
	var wg sync.WaitGroup

	for i, vec := range vectors {
		wg.Add(1)
		go func(idx int, vec []float32) {
			defer wg.Done()

			searchResults, err := w.client.Experimental().Search().
				WithCollection(w.collectionName).
				WithConsistencyLevel("ONE").
				WithLimit(topK).
				WithProperties(idPropName).
				WithMetadata(&wgraphql.Metadata{Distance: true}).
				WithNearVector(w.client.GraphQL().NearVectorArgBuilder().WithVector(vec)).
				Do(ctx)
			if err != nil {
				errCh <- err
				return
			}

			ids := make([]int64, 0, len(searchResults))
			scores := make([]float32, 0, len(searchResults))
			for _, hit := range searchResults {
				rawID, ok := hit.Properties[idPropName]
				if !ok {
					continue
				}
				switch value := rawID.(type) {
				case int64:
					ids = append(ids, value)
				case int:
					ids = append(ids, int64(value))
				case float64:
					ids = append(ids, int64(value))
				case float32:
					ids = append(ids, int64(value))
				default:
					errCh <- fmt.Errorf("unexpected %s type %T", idPropName, rawID)
					return
				}
				scores = append(scores, hit.Metadata.Distance)
			}

			results[idx] = queryResult{IDs: ids, Scores: scores}
		}(i, vec)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (w *weaviateBackend) Close(context.Context) error {
	return nil
}

func pickNodeTarget(r role, clientID int) (*nodeTarget, error) {
	registryPath := getenvDefault("WEAVIATE_REGISTRY_PATH", "./ip_registry.txt")
	_ = r
	_ = clientID
	return getNodeByRank(registryPath, 0)
}

func getNodeByRank(filename string, targetRank int) (*nodeTarget, error) {
	nodes, err := getAllNodes(filename)
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		if node.Rank == targetRank {
			return node, nil
		}
	}
	return nil, fmt.Errorf("rank %d not found in %s", targetRank, filename)
}

func getAllNodes(filename string) ([]*nodeTarget, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	nodes := make([]*nodeTarget, 0)
	for scanner.Scan() {
		// Expected format: rank,node,ip,http,grpc,...
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}
		rank, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		httpPort, err := strconv.Atoi(parts[3])
		if err != nil {
			return nil, err
		}
		grpcPort, err := strconv.Atoi(parts[4])
		if err != nil {
			return nil, err
		}
		nodes = append(nodes, &nodeTarget{
			Rank:     rank,
			Node:     parts[1],
			IP:       parts[2],
			HTTPPort: httpPort,
			GRPCPort: grpcPort,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return nodes, nil
}

func writeJSONL(path string, records any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	switch typed := records.(type) {
	case []insertLogRecord:
		// Records are written one JSON object per line to make downstream parsing streaming-friendly.
		for _, record := range typed {
			if err := enc.Encode(record); err != nil {
				return err
			}
		}
	case []queryLogRecord:
		for _, record := range typed {
			if err := enc.Encode(record); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported jsonl record type %T", records)
	}
	return nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
