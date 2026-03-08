# 🚀 从 C++ 到 Python：面向对象与核心机制对比学习笔记

> 本笔记基于 C++ 开发者的视角，通过代码实战对比，快速掌握 Python 的核心机制与地道写法。

---

## 1. 基础数据结构与 STL 对比

### 1.1 `Struct` 与 `std::vector`
*   **C++ (`struct` / `std::vector`)**: 需要显式声明类型，`struct` 是公开的值类型。
*   **Python (`@dataclass` / `List`)**:
    *   使用 `@dataclass` 装饰器自动生成 `__init__` (类似于构造函数)。
    *   使用内置的 `List` (`[]`) 替代 `vector`，支持混搭类型且自带动态扩容。

```python
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    score: int

students = [
    Student("Alice", 85),
    Student("Bob", 92)
]
# 排序 (等价于 std::sort + lambda)
students.sort(key=lambda s: s.score, reverse=True)
```

### 1.2 `std::map`、`std::set` 与 `std::string`
*   **Python `str`**: 字符串是**不可变**的，自带丰富的内置方法（如 `.lower()`, `.split()`）。
*   **Python `set`**: 原生支持哈希集合，使用 `{1, 2}` 初始化，支持数学运算 (`&` 交集, `|` 并集)。
*   **Python `dict` (字典)**: 替代 `std::unordered_map`，极其高效，使用 `{"a": 1}` 初始化。

```python
text = "Apple orange apple"
words = text.lower().split()

# Set 去重
unique_words = set(words)

# Dict 统计词频
word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1
```

---

## 2. 内存模型：指针、引用与深浅拷贝
这是 C++ 开发者最容易踩坑的地方！

*   **C++ (值语义)**: `a = b` 默认发生**深拷贝**，修改 `a` 不会影响 `b`。
*   **Python (引用语义)**: **一切皆对象，所有变量只是“标签”**。`a = b` 只是让两个标签贴在同一个内存对象上（类似隐式的 `std::shared_ptr`）。

```python
# 浅拷贝陷阱
a = [1, 2, 3]
b = a          # ⚠️ b 和 a 指向同一个列表！
b.append(4)    # a 也变成了 [1, 2, 3, 4]

# 正确的拷贝方式
b = a.copy()               # 浅拷贝 (只拷贝第一层)
import copy
matrix_c = copy.deepcopy(matrix_a)  # 深拷贝 (完全独立，等价于 C++ 的深拷贝)
```

---

## 3. 告别 for 循环：列表推导式
Python 提倡声明式编程，使用**推导式 (Comprehension)** 替代 C++ 的 `std::transform` 和 `std::copy_if`。

```python
nums = [1, 2, 3, 4, 5]

# 映射 (Map) -> std::transform
squares = [x * x for x in nums]

# 过滤 (Filter) -> std::copy_if
evens = [x for x in nums if x % 2 == 0]

# 映射 + 过滤
even_squares = [x * x for x in nums if x % 2 == 0]
```

---

## 4. 面向对象 (OOP) 的核心差异

### 4.1 `this` 指针 vs `self` 参数
*   **C++ (`this`)**: 编译器隐式传递的指针。
*   **Python (`self`)**: 显式传递。所有实例方法的第一个参数**必须**是 `self`，访问成员变量必须带上 `self.` 前缀（否则会被当成局部变量）。

### 4.2 构造函数 `__init__`
*   Python 的构造函数永远叫 `__init__`。它没有 `new` 关键字，直接调用类名即可实例化。**所有的实例变量必须在 `__init__` 中通过 `self.` 声明并初始化。**

### 4.3 访问控制 `public` / `private`
*   **C++**: 编译器强制的访问隔离。
*   **Python**: “防君子不防小人”。没有真正的 private，约定俗成**单下划线开头（如 `self._name`）表示受保护的内部变量**，外部不应直接访问。

### 4.4 继承与重写 (Override)
*   Python 天生支持多态（鸭子类型），不需要 `virtual` 或 `override` 关键字。
*   子类通过 `super().__init__()` 调用父类构造函数。

```python
class Dog:
    def __init__(self, name):
        self._name = name  # 约定为私有

    def speak(self):
        print(f"{self._name} 汪汪叫")

class FlyingDog(Dog):
    def __init__(self, name):
        super().__init__(name)  # 调用父类构造

    def speak(self):            # 直接覆盖，不需要 virtual
        print(f"{self._name} 在天上叫！")
```

---

## 5. LeetCode 实战：最小栈 (Min Stack)

对比 C++ 在 `private` 声明 `std::stack`，Python 直接在 `__init__` 中使用 `list` (`[]`) 模拟栈。

```python
class MinStack:
    def __init__(self):
        self.data_st = []  # 主栈
        self.min_st = []   # 最小值栈

    def push(self, val: int) -> None:
        self.data_st.append(val)
        if not self.min_st or val <= self.min_st[-1]:
            self.min_st.append(val)

    def pop(self) -> None:
        if self.data_st[-1] == self.min_st[-1]:
            self.min_st.pop()
        self.data_st.pop()

    def top(self) -> int:
        return self.data_st[-1] # [-1] 优雅获取栈顶

    def getMin(self) -> int:
        return self.min_st[-1]
```