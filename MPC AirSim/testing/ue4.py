"""用来确认如何用脚本控制ue4里的物体，本脚本无法控制actor在运行场景时持续移动"""

import unreal

def move_actor():
    # 获取关卡中的所有 Actors
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()

    # 定义目标 Actor 名称
    target_name = "OrangeBall"

    # 查找目标 Actor
    target_actor = None
    for actor in all_actors:
        if actor.get_name() == target_name:
            target_actor = actor
            break

    if target_actor:
        # 打印确认找到目标 Actor
        print(f"Found actor: {target_actor.get_name()}")

        # 获取当前位置
        current_location = target_actor.get_actor_location()

        # 更新位置：向上移动 100 单位
        new_location = unreal.Vector(current_location.x, current_location.y, current_location.z + 100)
        target_actor.set_actor_location(new_location)

        # 打印完成信息
        print(f"Moved actor '{target_name}' to new location: {new_location}")
    else:
        print(f"Actor '{target_name}' not found!")

# 执行函数
move_actor()

    