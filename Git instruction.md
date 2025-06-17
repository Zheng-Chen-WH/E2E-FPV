# Git相关指令
## 1. 设置SSH key
> **windows系统中右键打开git bash窗口才行，ubuntu直接打开终端就可以**
>
### windows设置：
**初始设置**   
```
git config --global user.name "Zheng-Chen-WH"    
git config --global user.email "ZhengchenWH@gmail.com"
```  
**生成SSH（如果没有）**  
```
ssh-keygen -t rsa -C "zhengchenWH@gmail.com"
```
连按三次回车，在C:\Users\Dendrobium\\.ssh\下生成私钥id_rsa和公钥id_rsa_pub     
**使用GitBash 将SSH私钥添加到 ssh-agent**  
```
eval $(ssh-agent -s)    
ssh-add /C:/Users/Dendrobium/.ssh/id_rsa (注意反斜杠和斜杠的区别)
```
**把SSH key添加到github**  
```
clip < /C:/Users/Dendrobium/.ssh/id_rsa.pub
```    
进入https://github.com/settings/keys ，新建ssh   
**测试SSH连接**  
```
ssh -T git@github.com
```  
点yes，成功的话会输出
```
Hi Zheng-Chen-WH! You've successfully authenticated, but GitHub does not provide shell access.
```

### Ubuntu设置（仅github）
**生成SSH**   
```
ssh-keygen -t rsa -b 4096 -C "zhengchenWH@gmail.com"
```   
**添加ssh私钥**    
```
eval "$(ssh-agent -s)"     
ssh-add ~/.ssh/id_rsa
```
**公钥添加到github（同上）**
**修改端口**   
在.ssh下创建config文件，内容为：  
```
Host github.com   
  Hostname ssh.github.com   
  Port 443    
```  

## 2. 基本设置
**初始化本地 Git 仓库（如果还没有）:** ```git init```  
**添加 GitHub SSH远程仓库**：
```
git remote add github git@github.com:Zheng-Chen-WH/E2E-FPV.git
``` 
（添加github SSH repository，名字为github）  
**添加 GitLab HTTP远程仓库（仅台式机）**：
```
git remote add gitlab http://gitlab.qypercep.com/Dendrobium/e2e-fpv.git
```
 （添加gitlab http repository，名字为gitlab）；gitlab难以识别SSH key所以用http  
**检查远程仓库**：
```
git remote -v
```  
**修改远程仓库链接**：
```
git remote "远程仓库名" set-url [url]
```  
**仓库克隆**：
```
git clone <URL> .
```
将代码直接克隆到你当前所在的空白文件夹中，而不是在其中创建新的子文件夹  

## 3. 日常协作
1. 在本地文件夹下启动终端（ubuntu）或git bash（windows）
2. 拉取最新更改：```git pull```，将远程仓库的更改下载到你的本地仓库，并尝试合并它们
    + ```git fetch```：拉取代码变更但不合并，将远程分支的最新状态下载到 ```origin/<branch_name>```，可以通过```git log origin/master```查看更改
    + ```git merge```：合并远程代码更改，```git merge origin/master```
3. 查看文件更改状态：```git status```
4. 暂存修改：```git add```     
    + 暂存所有更改（不包括新创建但未被 Git 追踪的文件）：```git add -u```
    + 暂存所有更改（包括新创建的文件）：```git add .```
    + 暂存特定文件：```git add path/to/your/file.py another_file.txt```
5. 提交更改，将暂存区的更改保存为本地仓库历史中的一个新版本，```git commit -m "你的提交消息，简明扼要地描述你做了什么"```
6. 推送本地更改：```git push```
    + **推送到两个仓库**：
    ```
    git push github master
    git push gitlab master 
    ```
    (随后输入账号Dendrobium，密码AAaa,,11)
7. push时产生冲突意味着本地文件和远程仓库在同一个文件的同一部分都做了修改。需要手动解决这些冲突，然后再次git add冲突文件，并git commit来完成合并。
## 4. 上传大文件（以模型pt文件为例）
```
# 对于 Ubuntu
sudo apt-get install git-lfs # 安装git-lfs
git lfs install # 初始化 Git LFS
git lfs track "*.pt" # 跟踪.pt类型文件
git add .gitattributes # .gitattributes文件自动生成并包含您所跟踪的文件类型
git commit -m "xxxx" # 提交更改
```

## 5. 重写历史（危险操作）
Git 会跟踪文件的历史记录，而不仅仅是当前工作目录的状态，因此历史记录中出现问题时，无法通过修改本地文件解决，必须将问题文件从Git历史中完全删除。 
> 警告：重写历史是具有破坏性的操作，因为它会改变提交的 SHA 值。如果仓库是多人协作的，必须与团队成员沟通，并确保他们拉取最新的更改。对于个人仓库，可以放心操作，但仍建议先备份。

**运行 git filter-branch**：
```
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch '要删除的文件的完整路径'" \
  --prune-empty --tag-name-filter cat -- --all
```
**清理旧的引用**：
```
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```
**强制推送到 GitHub**:
```
git push --force --all
git push --force --tags
```

