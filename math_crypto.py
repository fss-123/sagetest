# crypto_toolkit.py
# 90天国密算法工程师 - 主工具库（纯Python，任何环境可运行）
# 设计原则：极简 + 高性能 可无限拓展 注释清晰
from __future__ import annotations
from typing import Tuple, List, Optional
from math import gcd as _gcd
import random
from functools import reduce
import operator

# ==============================
# Day1：扩展欧几里得 + 模逆
# ==============================
def xgcd(a: int, b: int):
    """扩展欧几里得：返回 (g, x, y) 满足 a*x + b*y = g"""
    if a == 0:
        return b, 0, 1
    g, y, x = xgcd(b % a, a)
    return g, x - (b // a) * y, y


def modinv(a: int, m: int) -> int:
    """求模逆，要求 gcd(a,m)=1"""
    g, x, _ = xgcd(a, m)
    if g != 1:
        raise ValueError(f"No inverse: gcd({a},{m})={g}")
    return x % m


# ==============================
# Day2：欧拉函数 φ(n)
# ==============================
def euler_phi(n: int) -> int:
    """计算欧拉函数 φ(n)，支持任意正整数"""
    if n < 1: return 0
    res = n
    i = 2
    while i * i <= n:
        if n % i == 0:
            while n % i == 0:
                n //= i
            res -= res // i
        i += 1
    if n > 1:
        res -= res // n
    return res


# ==============================
# Day3：中国剩余定理（CRT）通用解法
# ==============================
def crt(congruences: List[Tuple[int, int]]) -> int:
    """
    中国剩余定理解同余方程组 x ≡ a_i (mod m_i)
    前提条件：所有 m_i 两两互质
    返回：唯一解 x (mod M)，M = m1*m2*...*mk
    """
    if not congruences:
        return 0

    # 计算总模数 M
    M = 1
    for _, m in congruences:  # 直接这样写！不要再套一层
        M *= m

    result = 0
    for a, m in congruences:
        Mi = M // m
        inv = modinv(Mi, m)  # 关键！昨天学的模逆
        result += a * Mi * inv

    return result % M


# ==============================
# Day4：Miller-Rabin 素性检测
# ==============================
def miller_rabin(n: int, witnesses: List[int] | int = None) -> bool:
    """
    Miller-Rabin 素性检测
    - witnesses 为列表 → 确定性检测
    - witnesses 为整数 k → 随机 k 轮概率检测（默认 k=40，错误率 < 2^-80）
    - 特殊情况快速判断
    """
    if n in {2, 3}:
        return True
    if n < 2 or n % 2 == 0:
        return False

    # 写 n-1 = 2^s * d
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # 确定性基底表（来自 Wikipedia/权威论文）
    deterministic = {
        1_000_000_000: [2, 7, 61],                    # n < 10^9
        3_317_044_064_679_887_385_961_981: [2,3,5,7,11,13,17],  # n < 3.3e18
        2**64: [2,3,5,7,11,13,17,23,29,31,37,41],   # 2048位够用
    }

    if witnesses is None:
        # 自动选择确定性基底
        for limit, bases in deterministic.items():
            if n < limit:
                witnesses = bases
                break
        else:
            witnesses = 40  # 超大数用随机40轮

    if isinstance(witnesses, int):  # 概率模式
        witnesses = [random.randint(2, n-2) for _ in range(witnesses)]

    for a in witnesses:
        if a >= n:
            continue
        if _gcd(a, n) != 1:
            return False

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
            if x == 1:
                return False  # 提前变1，必为合数
        else:
            return False  # 没出现 -1
    return True


# 测试1000以内所有素数
def primes_below_1000() -> List[int]:
    return [n for n in range(2, 1000) if miller_rabin(n)]


# ==============================
# Day5：有限域 GF(2⁸) 运算（模 0x11B = x⁸ + x⁴ + x³ + x + 1）
# ==============================
GF256_IRRED = 0x11B  # AES/SM4 共用的不可约多项式


def gf256_add(a: int, b: int) -> int:
    """GF(2⁸) 加法 = 按位异或"""
    return a ^ b


def gf256_mul(a: int, b: int) -> int:
    """GF(2⁸) 乘法，模 0x11B = x⁸+x⁴+x³+x+1（SM4/AES 标准）"""
    result = 0
    a = a & 0xFF
    b = b & 0xFF

    while b:
        if b & 1:  # b 的最低位是1
            result ^= a  # result += a (在 GF(2) 上就是 XOR)
        # 检查 a 的最高位是否为1（即第8位）
        if a & 0x80:
            a = (a << 1) ^ 0x11B  # 先左移，再减去多项式（XOR）
        else:
            a <<= 1  # 正常左移
        a &= 0xFF  # 强制截断为 8 位，防止溢出
        b >>= 1

    return result

# 可读性更高的版本
def gf256_mul_verbose(a: int, b: int) -> int:
    """GF(2⁸) 乘法（带注释版）"""
    p = 0
    for i in range(8):
        if b & (1 << i):
            p ^= (a << i)
    # 约简：最高位开始处理
    for i in range(14, 7, -1):  # x¹⁴⁴ 到 x⁸
        if p & (1 << i):
            p ^= GF256_IRRED << (i - 8)
    return p & 0xFF



# ==============================
# Day6：椭圆曲线点加（Weierstrass 形式 y² = x³ + ax + b）
# ==============================
class ECPoint:
    """椭圆曲线上的点（支持无穷远点 O）"""

    def __init__(self, x: Optional[int], y: Optional[int], curve) -> None:
        self.curve = curve
        if x is None and y is None:
            self.x = self.y = None
        else:
            p = curve[2]
            self.x = x % p if x is not None else None  # ←←←←← 加上这行！！
            self.y = y % p if y is not None else None  # 你原来就有

    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __eq__(self, other) -> bool:
        if not isinstance(other, ECPoint):
            return False
        if self.curve != other.curve:
            return False
        if self.is_infinity() or other.is_infinity():
            return self.is_infinity() and other.is_infinity()
        return self.x == other.x and self.y == other.y  # 直接比！因为我们保证了都在 [0,p-1]

    def __neg__(self):
        if self.is_infinity():
            return self
        p = self.curve[2]
        return ECPoint(self.x, -self.y % p, self.curve)  # 关键！-self.y % p 才是正确的负元

    def __add__(self, other):
        """点加主函数"""
        if self.is_infinity():
            return other
        if other.is_infinity():
            return self

        x1, y1 = self.x, self.y
        x2, y2 = other.x, other.y
        a, b, p = self.curve

        if x1 == x2 and (y1 + y2) % p == 0:  # P + (-P) = O
            return ECPoint(None, None, self.curve)

        if x1 == x2:  # 倍点
            lam = (3 * x1 * x1 + a) * modinv(2 * y1, p) % p
        else:  # 普通点加
            lam = (y2 - y1) * modinv(x2 - x1, p) % p

        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p  # 已经 %p 了
        return ECPoint(x3, y3, self.curve)  # __init__ 里还会再 % 一次，更安全

    def __repr__(self):
        if self.is_infinity():
            return "O (无穷远点)"
        return f"({self.x}, {self.y})"

# 自检用 toy 曲线：y² = x³ - 7x + 10 (mod 19)






# ==============================
# 自检（直接运行本文件自动验证前两天）
# ==============================
if __name__ == "__main__":
    print("crypto_toolkit 自检中...")

    # Day1
    assert modinv(17, 3120) == 2753
    assert modinv(77, 5) == 3

    # Day2
    assert euler_phi(21) == 12
    assert pow(2, 12, 21) == 1

    # Day3 CRT 经典例题全验证
    assert crt([(1, 3), (3, 5), (2, 7)]) == 58
    assert crt([(2, 3), (3, 5), (2, 7)]) == 23
    assert crt([(2, 5), (3, 7), (1, 11)]) == 122
    assert crt([(3, 5), (4, 7), (2, 11), (5, 13)]) == 3203

    # Day4 Miller-Rabin
    assert miller_rabin(13) == True
    assert miller_rabin(561) == False  # Carmichael 数被秒杀
    assert miller_rabin(10007) == True
    assert miller_rabin(91) == False

    primes = primes_below_1000()
    assert len(primes) == 168  # 1000以内有168个素数

    # Day5 GF(2⁸) 手算验证（已三重验证正确）
    assert gf256_add(0x57, 0x83) == 0xD4
    assert gf256_mul(0x57, 0x83) == 0xC1
    assert gf256_mul(0x57, 0x02) == 0xAE

    assert gf256_mul(0xCA, 0xFE) == 0x4C  # 正确答案是 0x4C
    assert gf256_mul(0xFF, 0xFF) == 0x13  # 正确答案是 0x13

    assert gf256_mul(0x00, 0x13) == 0x00
    assert gf256_mul(0x01, 0x01) == 0x01
    assert gf256_mul(0x02, 0x03) == 0x06

    # Day6 椭圆曲线点加验证（修正后正确答案）
    # TOY_CURVE = (0, 7, 17)  # y² = x³ + 7 (mod 17)
    #
    # P = ECPoint(2, 7, TOY_CURVE)
    # Q = ECPoint(3, 13, TOY_CURVE)  # ⚠️ 此点不在曲线上，仅作示例
    # R = P + Q
    # print(R)  # 输出 (14, 6)
    # assert R == ECPoint(14, 6, TOY_CURVE)  # ✅ 修改为实际计算结果
    #
    # # 倍点验证（同步修正）
    # assert P + P == ECPoint(12, 16, TOY_CURVE)  # ✅ P+P 的正确结果
    # Day6 椭圆曲线点加验证（使用有效点）
    TOY_CURVE = (0, 7, 17)  # y² = x³ + 7 (mod 17)

    # 验证有效的曲线点（手动计算验证过）
    P = ECPoint(2, 7, TOY_CURVE)  # 2³+7=15, 7²=49≡15 ✓
    Q = ECPoint(5, 8, TOY_CURVE)  # 5³+7=132≡13, 8²=64≡13 ✓

    R = P + Q
    print(R)  # 输出 (12, 1)
    assert R == ECPoint(12, 1, TOY_CURVE)  # ✅ 正确断言

    # 倍点验证
    assert P + P == ECPoint(12, 16, TOY_CURVE)  # ✅ P+P 的正确结果

    print(f"Day1-Day6 全部通过！1000 以内共有 {len(primes)} 个素数")