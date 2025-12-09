# sagetest
这是一个学习现代密码学的一个小的数学基础测试



项目结构
sagatest/
├── venv/                    # Python 虚拟环境
├── math_crypto.py          # 纯 Python 实现的密码学算法
└── sagetest.py             # SageMath 专用验证脚本


验证内容
所有算法均通过测试断言验证

GF(2⁸) 运算结果与 AES/SM4 标准一致

椭圆曲线点加运算正确性验证

依赖
Python 3.7+

SageMath（仅用于 sagetest.py）
