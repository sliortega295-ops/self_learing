# 🚀 从 C++ 到 Python：面向对象与核心机制对比学习笔记

> 本笔记基于 C++ 开发者的视角，通过代码实战对比，快速掌握 Python 的核心机制与地道写法。

---

## 1. 基础数据结构与 STL 对比

### 1.1 `Struct` 与 `std::vector`
*   **C++ (`struct` / `std::vector`)**: 需要显式声明类型，`struct` 是公开的值类型。

**C++ 基准代码**:
```cpp
#include <vector>
#include <string>
#include <algorithm>

struct Student {
    std::string name;
    int score;
};

int main() {
    std::vector<Student> students = {
        {"Alice", 85}, {"Bob", 92}
    };
    std::sort(students.begin(), students.end(), [](const Student& a, const Student& b) {
        return a.score > b.score;
    });
}
```

*   **Python (`@dataclass` / `List`)**:
    *   使用 `@dataclass` 装饰器自动生成 `__init__` (类似于构造函数)。
    *   使用内置的 `List` (`[]`) 替代 `vector`，支持混搭类型且自带动态扩容。

**Python 等价代码**:
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
*   **C++**: 使用 `<string>`, `<set>`, `<map>`，操作繁琐。

**C++ 基准代码**:
```cpp
#include <string>
#include <set>
#include <map>
#include <sstream>

std::string text = "Apple orange apple";
std::set<std::string> unique_words;
std::map<std::string, int> word_counts;

std::istringstream iss(text);
std::string word;
while (iss >> word) {
    unique_words.insert(word);
    word_counts[word]++;
}
```

*   **Python `str`**: 字符串是**不可变**的，自带丰富的内置方法（如 `.lower()`, `.split()`）。
*   **Python `set`**: 原生支持哈希集合，使用 `{1, 2}` 初始化，支持数学运算 (`&` 交集, `|` 并集)。
*   **Python `dict` (字典)**: 替代 `std::unordered_map`，极其高效，使用 `{"a": 1}` 初始化。

**Python 等价代码**:
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
这是 C++ 开发者最容易踩坑的地方！**Python 中没有 C++ 意义上的指针语法（`*` 和 `&`），但一切变量的本质都是“智能指针（对象引用）”。**

### 2.1 本质模型区别：盒子模型 vs 标签模型
*   **C++ (盒子模型 & 值语义)**: 当声明 `int a = 10;` 时，内存里挖了一个“盒子” `a` 装入 10。`a = b` 默认发生**深拷贝**，把 `b` 盒子里的东西复印一份装进 `a`，修改 `a` 不会影响 `b`。
*   **Python (标签模型 & 引用语义)**: **一切皆对象，所有变量只是贴在对象上的“标签”**。`a = b` 绝对不会发生复制！它只是让两个标签拴在同一个内存对象（气球）上（类似隐式的 `std::shared_ptr`）。

**Python 陷阱与拷贝**:
```python
# 浅拷贝陷阱 (标签拴在同一个列表对象上)
a = [1, 2, 3]
b = a          # ⚠️ b 和 a 指向同一个列表！
b.append(4)    # a 也变成了 [1, 2, 3, 4]

# 正确的拷贝方式
b = a.copy()               # 浅拷贝 (只拷贝第一层)
import copy
matrix_a = [[1, 2], [3, 4]]
matrix_c = copy.deepcopy(matrix_a)  # 深拷贝 (完全独立，等价于 C++ 的嵌套拷贝)
```

### 2.2 函数传参机制 (Pass-by-object-reference)
*   **C++**: 有传值（`int`）、传指针（`int*`）、传引用（`int&`）。
*   **Python**: **只有一种传参方式：传对象引用**。你可以理解为传入的参数是标签的一份复印件（拴着同一个气球）。

```python
def modify(lst):
    # 场景 1：通过标签修改气球内部 (会影响外部！)
    lst.append(99)

    # 场景 2：重绑定 (Rebinding) —— 把标签撕下来贴到新气球上
    # 这绝对不会影响外部的标签！(相当于 C++ 修改了局部指针的指向: p = new vector...)
    lst = [4, 5, 6]

my_list = [1, 2, 3]
modify(my_list)
print(my_list)  # 结果是 [1, 2, 3, 99]
```

### 2.3 可变 (Mutable) 与不可变 (Immutable) 对象
C++ 开发者常问：“既然赋值都是贴标签，为什么数字的赋值看起来像值拷贝？”
```python
a = 10
b = a
b = 20
print(a) # 输出 10！为什么 a 没变？
```
这是因为 Python 将对象分为两类：
*   **不可变对象 (Immutable)**：数字 (`int`, `float`)、字符串 (`str`)、元组 (`tuple`)。它们是“钢铁打的气球”，一旦造出就不能修改内部。当你写 `b = 20` 时，你不是在修改 10，而是**造了一个全新的气球 20**，然后把标签 `b` 移过去。标签 `a` 还在 10 上。
*   **可变对象 (Mutable)**：列表 (`list`)、字典 (`dict`)、集合 (`set`)、自定义类。它们是“橡胶打的气球”，可以通过方法（如 `.append()`）修改内部。修改内部时，所有拴着它的标签都会看到变化。

---

## 3. 告别 for 循环：列表推导式
Python 提倡声明式编程，使用**推导式 (Comprehension)** 替代 C++ 的 `std::transform` 和 `std::copy_if`。

**C++ 基准代码**:
```cpp
std::vector<int> nums = {1, 2, 3, 4, 5};
std::vector<int> even_squares;
for (int x : nums) {
    if (x % 2 == 0) {
        even_squares.push_back(x * x);
    }
}
```

**Python 等价代码**:
```python
nums = [1, 2, 3, 4, 5]

# 映射 (Map) -> std::transform
squares = [x * x for x in nums]

# 过滤 (Filter) -> std::copy_if
evens = [x for x in nums if x % 2 == 0]

# 映射 + 过滤 (终极一行流)
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
*   **C++**: 严格控制，需要 `virtual`、`public` 继承和初始化列表。

**C++ 基准代码**:
```cpp
class Dog {
protected:
    std::string name;
public:
    Dog(std::string n) : name(n) {}
    virtual void speak() { std::cout << name << " 汪汪叫\n"; }
};

class FlyingDog : public Dog {
public:
    FlyingDog(std::string n) : Dog(n) {}
    void speak() override { std::cout << name << " 在天上叫！\n"; }
};
```

*   **Python**: 天生支持多态（鸭子类型），不需要 `virtual` 或 `override` 关键字。子类通过 `super().__init__()` 调用父类构造函数。

**Python 等价代码**:
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

**C++ 基准代码**:
```cpp
class MinStack {
private:
    std::stack<int> data_st;
    std::stack<int> min_st;

public:
    MinStack() {}

    void push(int val) {
        data_st.push(val);
        if (min_st.empty() || val <= min_st.top()) {
            min_st.push(val);
        }
    }

    void pop() {
        if (data_st.top() == min_st.top()) min_st.pop();
        data_st.pop();
    }

    int top() { return data_st.top(); }
    int getMin() { return min_st.top(); }
};
```

**Python 等价代码**:
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