#include <iostream>
using namespace std;


class B {
    int a;
    public :
        B(int i = 2): a(i) {num++;cout<<a<<"_"<<num;}
        static int num;
    };
int B::num = 0;
void f3() {
    B o1;
    if(o1.num>3)
        throw 1.0;
    else
        throw -1;
}

void f2() {
    B *o2=new B;
    try {
        f3();
    } catch (double) {
        cout << "#";
    }
}

void f1() {
    try {
        f2();
        throw 1;
    } catch (int) {
        cout << "&" << endl;
    }
}

int main() {
    B o3(5);
    B &o4 = o3;
    f1();
    return 0;
}

// 确实可以 5_12_22_3&
// cd "c:\Users\13162\Desktop\工作\多模\leetcode\" && g++ test.cpp -o test && "c:\Users\13162\Desktop\工作\多模\leetcode\"test
// 可惜没有 debug

// 让我们逐步分析这段程序的输出。

// 1. 类 B 的定义：
// 类 B 有一个整数成员变量 a，构造函数带有默认值 i=2。
// 静态成员变量 num 共享在所有 B 对象中。
// 构造函数初始化 a，并将 num 递增，输出当前的 a 和 num。
// 2. B::num 静态变量的初始化：
// 在类外部，B::num 被初始化为 0。
// 3. 分析 main() 函数：
// B o3(5);：调用 B 的构造函数，参数为 5。此时 a 的值为 5，num 增加 1，num 变为 1。输出 5_1。
// B &o4 = o3;：创建一个 o3 的引用 o4，但不会调用构造函数，也不会输出。
// 4. f1() 调用：
// f1() 捕获了 f2() 抛出的异常，首先我们看 f2()。
// 5. f2() 调用：
// B *o2=new B;：通过动态分配内存创建一个 B 对象，调用构造函数，a 的值为 2（使用默认值），num 变为 2。输出 2_2。
// f2() 调用 f3()，并捕获 f3() 抛出的异常。
// 6. f3() 调用：
// B o1;：调用默认构造函数，a 的值为 2，num 变为 3。输出 2_3。
// if(o1.num>3)：此时 num 的值是 3，不满足条件 num > 3，所以执行 else 分支，抛出 -1（整数）。
// 7. 异常处理：
// f3() 抛出的 -1 被 f2() 忽略，因为它只捕获 double 类型的异常。
// f2() 返回给 f1()，f1() 捕获 int 类型的异常并输出 &。
// 总结输出顺序：
// 5_1（main() 中 B o3(5)）
// 2_2（f2() 中 new B）
// 2_3（f3() 中 B o1）
// &（f1() 捕获到 int 类型的异常）
// 因此，该程序的最终输出是：

// 复制代码
// 5_1 2_2 2_3 &



// The error you're encountering is because cout and endl belong to the std namespace, and they need to be properly declared or used with the std:: prefix. To fix this issue, either include using namespace std; at the top of your file or prefix cout and endl with std::.

// Here’s how you can modify your code:

// Option 1: Using using namespace std;
// cpp
// 复制代码
// #include <iostream>
// using namespace std;



