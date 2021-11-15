# 基于pytorch和flask的图像检索系统web示例

这是基于CNN图像特征抽取+相似度比对的图像检索系统实现

## 使用方式

通过以下命令安装工具库依赖

```
$ pip install requirements.txt
```

在models文件夹下有对应预训练模型的下载地址，大家可以下载后放置于models文件夹下

运行以下命令启动图像检索系统

```
$ python image_retrieval_main.py
```

如果是本地启动，可以通过: "http://127.0.0.1:8080/"访问网页版应用，如果是在远端服务器上运行，需要把127.0.0.1修改为对应的IP地址。

## 本地检索与排序

如果你只想在本地测试检索的过程，可以单独运行retrieval.py脚本:

```shell
$ python retrieval.py
```

排序后的结果会被打印出来。
