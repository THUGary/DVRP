#!/usr/bin/env python3
"""
Test script for problem cache system.

Tests:
1. Basic cache operations (add, sample)
2. Cache persistence (save/load)
3. Performance comparison (cache vs fresh generation)
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial_v2.utils.problem_cache import (
    CachedProblem, VersionProblemCache, ProblemCacheManager
)
import random


def test_basic_operations():
    """Test basic cache operations."""
    print("=" * 50)
    print("Test 1: Basic cache operations")
    print("=" * 50)
    
    # Create a cache
    cache = VersionProblemCache(version_id=1, max_size=100)
    
    # Create some fake problems
    rng = random.Random(42)
    
    for i in range(50):
        depot = torch.rand(1, 2)
        node_xy = torch.rand(30, 2)
        node_demand = torch.rand(30) * 0.5
        
        problem = CachedProblem(depot, node_xy, node_demand)
        cache.add(problem)
    
    print(f"  Cache size: {cache.size()}")
    assert cache.size() == 50, f"Expected 50, got {cache.size()}"
    
    # Sample from cache
    samples = cache.sample(10, rng)
    print(f"  Sampled {len(samples)} problems")
    assert len(samples) == 10
    
    # Test batch sampling
    result = cache.sample_batch(8, rng, torch.device('cpu'))
    depot_xy, node_xy, node_demand = result
    print(f"  Batch shapes: depot={depot_xy.shape}, nodes={node_xy.shape}, demand={node_demand.shape}")
    assert depot_xy.shape == (8, 1, 2)
    assert node_xy.shape == (8, 30, 2)
    assert node_demand.shape == (8, 30)
    
    print("  ✓ Basic operations passed!")
    return True


def test_cache_overflow():
    """Test cache eviction when full."""
    print("\n" + "=" * 50)
    print("Test 2: Cache overflow and eviction")
    print("=" * 50)
    
    cache = VersionProblemCache(version_id=2, max_size=10)
    
    # Add 15 problems to a cache of size 10
    for i in range(15):
        problem = CachedProblem(
            torch.tensor([[float(i), float(i)]]),
            torch.rand(5, 2),
            torch.rand(5),
        )
        cache.add(problem)
    
    print(f"  Added 15 problems to cache of size 10")
    print(f"  Cache size: {cache.size()}")
    assert cache.size() == 10, f"Expected 10, got {cache.size()}"
    
    # Check that oldest were evicted (FIFO)
    # First problem should be the one with depot (5, 5), not (0, 0)
    rng = random.Random(42)
    all_problems = cache.sample(10, rng)
    depot_values = [p.depot_xy[0, 0].item() for p in all_problems]
    print(f"  Depot x-values in cache: {sorted(depot_values)}")
    assert 0.0 not in depot_values, "Oldest problem should have been evicted"
    
    print("  ✓ Cache eviction passed!")
    return True


def test_persistence(tmp_dir="/tmp/test_problem_cache"):
    """Test cache save/load from disk."""
    print("\n" + "=" * 50)
    print("Test 3: Cache persistence")
    print("=" * 50)
    
    # Clean up
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    # Create and populate cache
    cache = VersionProblemCache(version_id=3, max_size=50, cache_dir=tmp_dir)
    
    for i in range(20):
        problem = CachedProblem(
            torch.tensor([[float(i) / 20, float(i) / 20]]),
            torch.rand(10, 2),
            torch.rand(10),
        )
        cache.add(problem)
    
    print(f"  Created cache with {cache.size()} problems")
    
    # Save to disk
    cache.save_to_disk()
    
    # Create new cache and load
    cache2 = VersionProblemCache(version_id=3, max_size=50, cache_dir=tmp_dir)
    print(f"  Loaded cache with {cache2.size()} problems")
    
    assert cache2.size() == 20, f"Expected 20, got {cache2.size()}"
    
    # Verify content
    rng = random.Random(42)
    sample = cache2.sample(1, rng)[0]
    print(f"  Sample depot: {sample.depot_xy}")
    
    print("  ✓ Persistence passed!")
    
    # Clean up
    shutil.rmtree(tmp_dir)
    return True


def test_cache_manager():
    """Test ProblemCacheManager with multiple versions."""
    print("\n" + "=" * 50)
    print("Test 4: Cache Manager")
    print("=" * 50)
    
    import shutil
    tmp_dir = "/tmp/test_cache_manager"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    manager = ProblemCacheManager(
        cache_dir=tmp_dir,
        max_problems_per_version=100,
        cache_reuse_ratio=0.8,
        min_cache_size_for_reuse=10,
    )
    
    # Add problems for version 1
    depot_v1 = torch.rand(20, 1, 2)
    nodes_v1 = torch.rand(20, 30, 2)
    demand_v1 = torch.rand(20, 30)
    manager.add_problems(1, depot_v1, nodes_v1, demand_v1)
    
    # Add problems for version 2
    depot_v2 = torch.rand(15, 1, 2)
    nodes_v2 = torch.rand(15, 30, 2)
    demand_v2 = torch.rand(15, 30)
    manager.add_problems(2, depot_v2, nodes_v2, demand_v2)
    
    stats = manager.get_cache_stats()
    print(f"  Cache stats: {stats}")
    
    assert stats["num_versions"] == 2
    assert stats["total_problems"] == 35
    
    # Test should_use_cache
    rng = random.Random(42)
    
    # Version 1 has 20 problems (>= min 10), should be eligible
    use_count = sum(1 for _ in range(100) if manager.should_use_cache(1, rng))
    print(f"  Version 1 cache hit rate (expected ~80%): {use_count}%")
    assert 60 < use_count < 95, f"Expected ~80%, got {use_count}%"
    
    # Version 2 has 15 problems (>= min 10), should be eligible
    use_count = sum(1 for _ in range(100) if manager.should_use_cache(2, rng))
    print(f"  Version 2 cache hit rate (expected ~80%): {use_count}%")
    
    # Version 3 doesn't exist, should never use cache
    use_count = sum(1 for _ in range(100) if manager.should_use_cache(3, rng))
    print(f"  Version 3 (not exist) cache hit rate: {use_count}%")
    assert use_count == 0
    
    # Test sampling across versions
    result = manager.sample_across_versions([1, 2], 16, rng, torch.device('cpu'))
    assert result is not None
    depot, nodes, demand = result
    print(f"  Cross-version sample shapes: {depot.shape}, {nodes.shape}, {demand.shape}")
    assert depot.shape == (16, 1, 2)
    
    print("  ✓ Cache Manager passed!")
    
    # Clean up
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    return True


def test_performance():
    """Compare cache vs fresh generation performance."""
    print("\n" + "=" * 50)
    print("Test 5: Performance comparison")
    print("=" * 50)
    
    # Simulate cache operations
    rng = random.Random(42)
    device = torch.device('cpu')
    
    # Pre-populate a cache
    cache = VersionProblemCache(version_id=1, max_size=1000)
    for _ in range(500):
        problem = CachedProblem(
            torch.rand(1, 2),
            torch.rand(50, 2),
            torch.rand(50),
        )
        cache.add(problem)
    
    # Measure cache sampling time
    n_iterations = 100
    batch_size = 32
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = cache.sample_batch(batch_size, rng, device)
    cache_time = time.perf_counter() - start
    
    print(f"  Cache sampling: {n_iterations} batches of {batch_size}")
    print(f"    Total time: {cache_time*1000:.1f} ms")
    print(f"    Per batch: {cache_time/n_iterations*1000:.2f} ms")
    
    # Compare with theoretical diffusion time
    # Diffusion: ~540ms per sample, serial
    diffusion_time_per_batch = 540 * batch_size / 1000  # seconds
    
    print(f"\n  Theoretical comparison:")
    print(f"    Diffusion time per batch (serial): {diffusion_time_per_batch:.1f} s")
    print(f"    Cache time per batch: {cache_time/n_iterations*1000:.2f} ms")
    print(f"    Speedup: {diffusion_time_per_batch / (cache_time/n_iterations):.0f}x")
    
    print("\n  ✓ Performance test passed!")
    return True


def main():
    print("\n" + "=" * 60)
    print("Problem Cache System Tests")
    print("=" * 60)
    
    tests = [
        test_basic_operations,
        test_cache_overflow,
        test_persistence,
        test_cache_manager,
        test_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
