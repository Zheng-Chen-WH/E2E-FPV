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
**删除远程仓库链接**：
```
git remote rm xxxxx
```
**仓库克隆**：
```
git clone <URL> .
```
将代码直接克隆到你当前所在的空白文件夹中，而不是在其中创建新的子文件夹  

## 3. 日常协作
1. 在本地文件夹下启动终端（ubuntu）或git bash（windows）
2. 拉取最新更改：```git pull github master```，将远程仓库的更改下载到你的本地仓库，并尝试合并它们
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
## 6.本地和远程仓库同时修改且二选一

1. 比较本地与远程的差异
```
git status
git fetch github # 拉取远程版本的最新改动，而不直接合并
git --no-pager diff # 不使用分页器
```

2. 保留本地版本：
```
# 推送本地更改 
git push --force github master
```

3. 保留远程版本
```
git reset --hard github/master #重置本地文件到远程版本
```

4. 处理未跟踪的文件
```
git clean -n # 查看这些文件并决定是否删除
git clean -f # 删除未跟踪文件
```

## 7.本地和远程仓库同时修改且需要合并

1. 提交本地修改
```
git add .
git commit -m "xxx"
```

2. 合并远程修改
```
git pull github master
```

3. 修改
  1. 打开提示有冲突的文件。
  2. 你会看到类似 <<<<<<< HEAD，=======，>>>>>>> 这样的标记。
  3. 你需要手动编辑文件，删除这些标记，并决定保留哪些代码。
  4. 保存文件后，使用 git add . 将解决后的文件标记为“已解决”。
  5. 最后，运行 git commit 来完成这次合并。

4. 再次推送本地代码

## 8.忽略所有Pycache

1. 在项目根目录创建或修改 .gitignore 文件
```
gitbash
touch .gitignore
```

2. 编辑.gitignore

```
# 忽略所有名为 __pycache__ 的文件夹
__pycache__/ 
```

## 9. 多branch协作

+ 查看你的分支情况
```
# 查看所有本地分支（当前分支前有 * 号）
git branch

# 查看所有远程分支
git branch -r

# 查看当前状态和所在分支
git status
```

+ 把当前分支推送到远程的 cz 分支
```
# 如果远程不存在 cz 分支，Git 会自动创建
git push github HEAD:cz
```
+ 创建一个新的本地分支 cz 并推送
```
# 创建并切换到 cz 分支（基于当前分支）
git checkout -b cz

# 推送到远程
git push github cz

# 如果本地是master:
git push github master:cz
```
+ 创建并切换到本地 cz 分支       
git fetch会下载远程的更新，并存储在远程跟踪分支（如 github/cz）中，但不会自动合并到工作目录
```
git fetch github # 更新远程信息
git checkout --track github/cz
```

+ 本地在master下存在不需要的修改，阻碍了切换分支：
```
# 把当前修改"藏起来"
git stash

# 切换分支
git checkout cz

# 在 cz 分支上恢复刚才的修改（如果需要）
git stash pop
# 如果不想恢复这些修改，直接 git stash drop 丢弃即可。
```