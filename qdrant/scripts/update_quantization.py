"""Update quantization config on an existing collection and wait for GREEN."""
import json
import os
import time
import urllib.request
from qdrant_client import QdrantClient, models


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_ip_from_file(filepath):
    with open(filepath) as f:
        rank, ip, port = f.readline().strip().split(",")
    return ip, int(port)


def build_quantization_config():
    quantization_type = os.getenv("QUANTIZATION_TYPE", "NONE").strip().upper()
    always_ram = is_truthy(os.getenv("QUANTIZATION_ALWAYS_RAM"))

    match quantization_type:
        case "NONE" | "":
            return None
        case "SCALAR":
            quantile_raw = os.getenv("QUANTIZATION_SCALAR_QUANTILE", "").strip()
            quantile = float(quantile_raw) if quantile_raw else None
            if quantile is not None and not 0 < quantile <= 1:
                raise ValueError("QUANTIZATION_SCALAR_QUANTILE must be in the range (0, 1]")
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=quantile,
                    always_ram=always_ram,
                ),
            )
        case "BINARY":
            encoding_name = os.getenv(
                "QUANTIZATION_BINARY_ENCODING", "DEFAULT"
            ).strip().upper()
            encoding = None
            if encoding_name != "DEFAULT":
                encoding = getattr(models.BinaryQuantizationEncoding, encoding_name)
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    encoding=encoding,
                    always_ram=always_ram,
                ),
            )
        case "PRODUCT":
            compression_name = os.getenv(
                "QUANTIZATION_PRODUCT_COMPRESSION", "X16"
            ).strip().upper()
            return models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=getattr(models.CompressionRatio, compression_name),
                    always_ram=always_ram,
                ),
            )
        case "TURBO":
            bits_name = os.getenv("QUANTIZATION_TURBO_BITS", "BITS4").strip().upper()
            return models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=getattr(models.TurboQuantBitSize, bits_name),
                    always_ram=always_ram,
                ),
            )
        case _:
            raise ValueError(f"Unknown quantization type: {quantization_type}")


def wait_for_green(host: str, rest_port: int, collection_name: str, timeout: int = 3600) -> None:
    url = f"http://{host}:{rest_port}/collections/{collection_name}"
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                info = json.load(resp)
            if info.get("result", {}).get("status") == "green":
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(
        f"collection {collection_name!r} did not become GREEN within {timeout}s"
    )


def main() -> None:
    base_ip, file_port = load_ip_from_file("ip_registry.txt")
    rest_port = file_port - 2
    grpc_port = rest_port + 1

    collection_name = os.environ["COLLECTION_NAME"].strip()
    quantization_type = os.getenv("QUANTIZATION_TYPE", "NONE").strip().upper()

    if quantization_type in ("NONE", ""):
        # Use raw REST PATCH so quantization_config is explicitly set to null.
        url = f"http://{base_ip}:{rest_port}/collections/{collection_name}"
        body = json.dumps({"quantization_config": None}).encode()
        req = urllib.request.Request(
            url, data=body, method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.load(resp)
        if not result.get("result"):
            raise RuntimeError(f"failed to disable quantization: {result}")
    else:
        quantization_config = build_quantization_config()
        client = QdrantClient(
            host=base_ip,
            port=rest_port,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=600,
            grpc_options={"grpc.enable_http_proxy": 0},
        )
        client.update_collection(
            collection_name=collection_name,
            quantization_config=quantization_config,
        )

    wait_for_green(base_ip, rest_port, collection_name)
    print(
        f"Updated quantization to {quantization_type} for {collection_name!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
