"""运行所有测试"""
import sys
import os
import unittest


def run_all():
    """发现并运行 tests/ 目录下所有 test_*.py 文件"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(test_dir, ".."))

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
