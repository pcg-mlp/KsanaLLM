为了便于在分布式环境下部署服务，使用两个脚本来打包和解包安装所需的文件。

# 前提设置
工程目录下有build目录，在build目录中进行cmake编译操作

pkg.sh用于打包相关文件，生成build/package.tar。

unpkg.sh用于解包package.tar，会在package.tar所在的目录下建一个KsanaLLM_runtime目录，然后把package.tar的内容解压到该目录。

因为编译中有一些路径依赖，所以pkg.sh和unpkg.sh中需要使用相同的路径。
