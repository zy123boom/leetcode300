package leetcode300_mutithread;

import java.util.concurrent.CountDownLatch;

/**
 * LeetCode 多线程题目
 * @author boomzy
 * @date 2020/2/3 18:49
 */
public class Main {

    /**
     *  LeetCode.1114 按序打印
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
}
