## HelloWorld自定义算子样例说明
<!--注：该样例仅用于说明目的，不用作生产质量代码的示例-->
本样例通过使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，核函数内通过printf打印输出结果。

## 支持的产品型号
样例支持的产品型号为：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品
- Atlas 200/500 A2推理产品

## 目录结构介绍
```
├── 0_helloworld
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── hello_world.cpp         // 算子kernel实现
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```

## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1.准备：获取样例代码

编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2.编译运行样例工程

  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/0_helloworld
    ```

  - 配置修改
    * 修改CMakeLists.txt内SOC_VERSION为所需产品型号
    * 修改CMakeLists.txt内ASCEND_CANN_PACKAGE_PATH为CANN包的安装路径

  - 样例执行

    ```bash
    bash run.sh -v [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行
      npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品
      - Atlas 200/500 A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -v Ascendxxxyy
    ```

## 更新说明
| 时间       | 更新事项                                     |
| ---------- | -------------------------------------------- |
| 2023/10/23 | 新增HelloWorldSample样例                     |
| 2024/03/07 | 修改样例编译方式                             |
| 2024/05/16 | 修改readme结构，新增目录结构                 |
| 2024/11/11 | 样例目录调整 |

## 已知issue

  暂无
