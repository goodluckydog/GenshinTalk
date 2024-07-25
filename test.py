import asyncio


# 定义一个异步生成器
async def async_generator():
    for i in range(10):
        await asyncio.sleep(1)  # 模拟异步操作
        yield i  # 产生一个值


# 定义主协程，它使用异步生成器
async def main():
    # 使用 async for 循环来迭代异步生成器产生的值
    async for value in async_generator():
        print(value)  # 打印每个接收到的值


# 运行主协程
asyncio.run(main())
