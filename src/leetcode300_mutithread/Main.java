package leetcode300_mutithread;

import java.util.concurrent.CountDownLatch;

/**
 * LeetCode 多线程题目
 *
 * @author boomzy
 * @date 2020/2/3 18:49
 */
public class Main {

    /**
     * LeetCode.1114 按序打印
     * <p>
     * 我们提供了一个类：
     * <p>
     * public class Foo {
     *   public void one() { print("one"); }
     *   public void two() { print("two"); }
     *   public void three() { print("three"); }
     * }
     * 三个不同的线程将会共用一个 Foo 实例。
     * <p>
     * 线程 A 将会调用 one() 方法
     * 线程 B 将会调用 two() 方法
     * 线程 C 将会调用 three() 方法
     * 请设计修改程序，以确保 two() 方法在 one() 方法之后被执行，three() 方法在 two() 方法之后被执行。
     * <p>
     *  
     * 示例 1:
     * 输入: [1,2,3]
     * 输出: "onetwothree"
     * 解释:
     * 有三个线程会被异步启动。
     * 输入 [1,2,3] 表示线程 A 将会调用 one() 方法，线程 B 将会调用 two() 方法，线程 C 将会调用 three() 方法。
     * 正确的输出是 "onetwothree"。
     * <p>
     * 示例 2:
     * 输入: [1,3,2]
     * 输出: "onetwothree"
     * 解释:
     * 输入 [1,3,2] 表示线程 A 将会调用 one() 方法，线程 B 将会调用 three() 方法，线程 C 将会调用 two() 方法。
     * 正确的输出是 "onetwothree"。
     */
    class Foo {

        // second的计数器
        private CountDownLatch c2;
        // third的计数器
        private CountDownLatch c3;

        public Foo() {
            c2 = new CountDownLatch(1);
            c3 = new CountDownLatch(1);
        }

        public void first(Runnable printFirst) throws InterruptedException {

            // printFirst.run() outputs "first". Do not change or remove this line.
            printFirst.run();
            c2.countDown();
        }

        public void second(Runnable printSecond) throws InterruptedException {

            c2.await();
            // printSecond.run() outputs "second". Do not change or remove this line.
            printSecond.run();
            c3.countDown();
        }

        public void third(Runnable printThird) throws InterruptedException {

            c3.await();
            // printThird.run() outputs "third". Do not change or remove this line.
            printThird.run();
        }
    }

    /**
     * LeetCode.1115 交替打印FooBar
     * <p>
     * 我们提供一个类：
     * <p>
     * class FooBar {
     *     public void foo() {
     *         for (int i = 0; i < n; i++) {
     *           print("foo");
     *         }
     *     }
     *     public void bar() {
     *         for (int i = 0; i < n; i++) {
     *         print("bar");
     *         }
     *     }
     * }
     * 两个不同的线程将会共用一个 FooBar 实例。其中一个线程将会调用 foo() 方法，另一个线程将会调用 bar() 方法。
     * 请设计修改程序，以确保 "foobar" 被输出 n 次。
     * <p>
     * 示例 1:
     * 输入: n = 1
     * 输出: "foobar"
     * 解释: 这里有两个线程被异步启动。其中一个调用 foo() 方法, 另一个调用 bar() 方法，"foobar" 将被输出一次。
     * <p>
     * 示例 2:
     * 输入: n = 2
     * 输出: "foobarfoobar"
     * 解释: "foobar" 将被输出两次。
     */
    class FooBar {
        private int n;
        private Object locker;
        // 计数器，奇数打"foo"，偶数打"bar"
        private volatile int cnt;

        public FooBar(int n) {
            this.n = n;
            this.locker = new Object();
            cnt = 0;
        }

        public void foo(Runnable printFoo) throws InterruptedException {

            for (int i = 0; i < n; i++) {
                while (true) {
                    synchronized (locker) {
                        if (cnt % 2 == 0) {
                            // printFoo.run() outputs "foo". Do not change or remove this line.
                            printFoo.run();
                            cnt++;
                            locker.notifyAll();
                            break;
                        } else {
                            locker.wait();
//                            locker.notifyAll();
                        }
                    }
                }
            }
        }

        public void bar(Runnable printBar) throws InterruptedException {

            for (int i = 0; i < n; i++) {
                while (true) {
                    synchronized (locker) {
                        if (cnt % 2 == 1) {
                            // printBar.run() outputs "bar". Do not change or remove this line.
                            printBar.run();
                            cnt++;
                            locker.notifyAll();
                            break;
                        } else {
                            locker.wait();
//                            locker.notifyAll();
                        }
                    }
                }
            }
        }
    }
}
