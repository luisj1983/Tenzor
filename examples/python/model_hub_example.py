#!/usr/bin/env python3
"""
Example demonstrating Tenzor ModelHub for downloading and using pretrained weights.

This example shows how to:
1. Configure the ModelHub
2. Download pretrained models
3. Load weights into a model
4. Use custom progress callbacks
5. Manage cache
"""

import tenzor as tz
import os

# Initialize Tenzor library
tz.initialize()

def format_size(bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def progress_callback(downloaded, total, speed, eta):
    """Custom progress callback for downloads."""
    if total > 0:
        percent = 100.0 * downloaded / total
        print(f"\rDownload: {percent:.1f}% ({format_size(downloaded)}/{format_size(total)}) "
              f"@ {format_size(speed)}/s - ETA: {eta:.0f}s", end='', flush=True)
    else:
        print(f"\rDownloaded: {format_size(downloaded)} @ {format_size(speed)}/s",
              end='', flush=True)

def example_basic_usage():
    """Example 1: Basic model download and loading."""
    print("=" * 70)
    print("Example 1: Basic Model Download and Loading")
    print("=" * 70)

    # List available registered models
    models = tz.models.Hub.list_registered_models()
    print(f"\nAvailable pretrained models: {len(models)}")
    print(f"Sample models: {models[:5]}")

    # Check if ResNet-50 is available
    if tz.models.Hub.is_registered("resnet50"):
        print("\nResNet-50 is registered in the hub")
        info = tz.models.Hub.get_model_info("resnet50")
        print(f"  URL: {info.url}")
        print(f"  Description: {info.description}")

        # Check if already cached
        if tz.models.Hub.is_cached("resnet50"):
            print("  Status: Already cached")
            path = tz.models.Hub.get_cached_path("resnet50")
            print(f"  Cached at: {path}")
        else:
            print("  Status: Not cached (would download on first use)")

    print()

def example_custom_model():
    """Example 2: Download custom model from URL."""
    print("=" * 70)
    print("Example 2: Download Custom Model")
    print("=" * 70)

    # Register a custom model
    custom_model = tz.models.ModelWeightInfo()
    custom_model.name = "my_custom_model"
    custom_model.url = "https://example.com/my_model.pth"
    custom_model.sha256 = "abcd1234..."  # Optional: for verification
    custom_model.description = "My custom pretrained model"

    tz.models.Hub.register_model(custom_model)
    print(f"Registered custom model: {custom_model.name}")
    print(f"  URL: {custom_model.url}")

    # Note: Actual download would happen here if URL was valid
    # weights_path = tz.models.Hub.download_pretrained("my_custom_model")

    print()

def example_config_management():
    """Example 3: Configure ModelHub settings."""
    print("=" * 70)
    print("Example 3: ModelHub Configuration")
    print("=" * 70)

    # Get current config
    config = tz.models.Hub.get_config()
    print(f"\nCurrent cache directory: {config.cache_dir}")
    print(f"Max cache size: {config.max_cache_size} bytes " +
          f"({'unlimited' if config.max_cache_size == 0 else format_size(config.max_cache_size)})")
    print(f"Verify checksums: {config.verify_checksums}")
    print(f"Resume downloads: {config.resume_downloads}")
    print(f"Connection timeout: {config.connection_timeout}s")
    print(f"Max retries: {config.max_retries}")

    # Modify config
    config.max_cache_size = 1024 * 1024 * 1024  # 1 GB
    config.connection_timeout = 60  # 60 seconds
    tz.models.Hub.set_config(config)
    print("\nUpdated configuration:")
    print(f"  Max cache size: {format_size(config.max_cache_size)}")
    print(f"  Connection timeout: {config.connection_timeout}s")

    print()

def example_cache_management():
    """Example 4: Manage cache."""
    print("=" * 70)
    print("Example 4: Cache Management")
    print("=" * 70)

    # Get cache info
    cache_dir = tz.models.Hub.get_cache_dir()
    cache_size = tz.models.Hub.cache_size()
    cached_models = tz.models.Hub.list_cached_models()

    print(f"\nCache directory: {cache_dir}")
    print(f"Cache size: {format_size(cache_size)}")
    print(f"Cached models: {len(cached_models)}")

    if cached_models:
        print("\nCached model list:")
        for model_name in cached_models[:5]:  # Show first 5
            path = tz.models.Hub.get_cached_path(model_name)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  - {model_name}: {format_size(size)}")

    # Clean cache (remove oldest files if over limit)
    # max_size = 500 * 1024 * 1024  # 500 MB
    # removed = tz.models.Hub.clean_cache(max_size)
    # print(f"\nCleaned cache: removed {removed} files")

    print()

def example_download_stats():
    """Example 5: Download statistics."""
    print("=" * 70)
    print("Example 5: Download Statistics")
    print("=" * 70)

    # After a download, you can get stats
    # This example shows the API, but won't actually download
    print("\nDownload stats (from last download):")

    try:
        stats = tz.models.Hub.get_last_download_stats()
        print(f"  Total bytes: {format_size(stats.total_bytes)}")
        print(f"  Downloaded: {format_size(stats.bytes_downloaded)}")
        print(f"  Time: {stats.download_time:.2f}s")
        print(f"  Average speed: {format_size(stats.average_speed)}/s")
        print(f"  Resumed: {stats.resumed}")
        print(f"  Verified: {stats.verified}")
    except Exception as e:
        print(f"  No download stats available yet: {e}")

    print()

def example_checksum_verification():
    """Example 6: Checksum verification."""
    print("=" * 70)
    print("Example 6: Checksum Verification")
    print("=" * 70)

    # Get cache directory
    cache_dir = tz.models.Hub.get_cache_dir()
    print(f"\nCache directory: {cache_dir}")

    # List cached files and compute checksums
    cached_models = tz.models.Hub.list_cached_models()

    if cached_models:
        print("\nComputing checksums for cached models:")
        for model_name in cached_models[:2]:  # Check first 2 models
            path = tz.models.Hub.get_cached_path(model_name)
            if os.path.exists(path):
                try:
                    checksum = tz.models.Hub.compute_checksum(path)
                    print(f"  {model_name}:")
                    print(f"    SHA256: {checksum}")

                    # Verify against known checksum (if available)
                    if tz.models.Hub.is_registered(model_name):
                        info = tz.models.Hub.get_model_info(model_name)
                        if info.sha256:
                            verified = tz.models.Hub.verify_checksum(path, info.sha256)
                            print(f"    Verified: {verified}")
                except Exception as e:
                    print(f"  {model_name}: Error - {e}")
    else:
        print("\nNo cached models available for checksum verification")

    print()

def example_load_into_model():
    """Example 7: Load weights into a model."""
    print("=" * 70)
    print("Example 7: Load Pretrained Weights into Model")
    print("=" * 70)

    # This example demonstrates the API
    # In practice, you would have an actual model instance

    print("\nExample code:")
    print("""
    # Create a model (e.g., ResNet-50)
    model = MyResNet50()

    # Download and load pretrained weights
    tz.models.load_pretrained(
        model,
        "resnet50",
        show_progress=True,
        strict=True  # Strict mode: fail if architecture mismatch
    )

    # Or download separately and load
    weights_path = tz.models.Hub.download_pretrained("resnet50")
    tz.models.Hub.load_pretrained_weights(model, weights_path, strict=False)
    """)

    print("\nNote: Actual model loading requires a compatible model architecture")
    print()

def example_custom_progress():
    """Example 8: Custom progress callback."""
    print("=" * 70)
    print("Example 8: Custom Progress Callback")
    print("=" * 70)

    print("\nExample of using custom progress callback:")
    print("""
    def my_progress(downloaded, total, speed, eta):
        percent = 100.0 * downloaded / total if total > 0 else 0
        print(f"Progress: {percent:.1f}% - Speed: {speed/1024:.2f} KB/s")

    # Download with custom progress callback
    weights_path = tz.models.Hub.download_pretrained(
        "resnet50",
        show_progress=False,  # Disable default progress
        progress_callback=my_progress  # Use custom callback
    )
    """)

    print()

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Tenzor ModelHub Examples")
    print("=" * 70 + "\n")

    # Initialize Tenzor
    tz.initialize()

    # Run examples
    example_basic_usage()
    example_custom_model()
    example_config_management()
    example_cache_management()
    example_download_stats()
    example_checksum_verification()
    example_load_into_model()
    example_custom_progress()

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
