from graphviz import Digraph

def draw_thesis_diagram():
    # 创建有向图，设置从上到下的布局，设置中文支持字体
    # 注意：Windows下常用 'SimHei' 或 'Microsoft YaHei'，Linux/Mac可能需要改为系统有的字体如 'Heiti TC'
    dot = Digraph(comment='AD Diagnosis Framework', format='png')
    dot.attr(rankdir='TB', fontname='SimHei', splines='ortho')
    
    # 全局节点属性设置（方框，字体大小等）
    dot.attr('node', shape='box', fontname='SimHei', style='rounded', fontsize='12')

    # === 1. 数据来源与预处理 ===
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='第一阶段：数据获取与预处理', fontname='SimHei', fontsize='14', style='dashed')
        c.node('ADNI', 'ADNI 数据库', shape='cylinder', style='filled', fillcolor='#E1F5FE')
        c.node('Preprocess', '数据预处理\n(FreeSurfer, ANTs, FSL)', style='filled', fillcolor='#F5F5F5')
        c.node('Data_Paired', '配对数据\n(MRI + PET)', shape='parallelogram', style='filled', fillcolor='#FFF9C4')
        c.node('Data_Unpaired', '未配对数据\n(MRI)', shape='parallelogram', style='filled', fillcolor='#FFF9C4')
        
        c.edge('ADNI', 'Preprocess')
        c.edge('Preprocess', 'Data_Paired')
        c.edge('Preprocess', 'Data_Unpaired')

    # === 2. 核心模型部分 ===
    # 使用一个大子图包裹两个并行的网络
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='第二阶段：核心网络构建', fontname='SimHei', fontsize='14', color='none')

        # --- 左侧：I2SB 生成网络 ---
        with c.subgraph(name='cluster_i2sb') as i2sb:
            i2sb.attr(label='模块A：基于 I2SB 的跨模态生成 (MRI -> PET)', style='rounded,filled', color='#E8F5E9')
            i2sb.node('Concat', '通道拼接输入\n(Concatenation)')
            i2sb.node('UNet', 'U-Net 生成网络', style='filled', fillcolor='#C8E6C9')
            i2sb.node('MSE', 'MSE Loss', shape='diamond', style='filled', fillcolor='#FFCDD2')
            i2sb.node('Gen_PET', '生成伪 PET', shape='parallelogram')
            
            i2sb.edge('Concat', 'UNet')
            i2sb.edge('UNet', 'Gen_PET')
            i2sb.edge('Gen_PET', 'MSE', style='dashed', label=' 计算损失')

        # --- 右侧：多模态分类网络 ---
        with c.subgraph(name='cluster_cls') as cls:
            cls.attr(label='模块B：多模态分类网络', style='rounded,filled', color='#E3F2FD')
            
            # 双流入口
            cls.node('Input_M', 'MRI 输入')
            cls.node('Input_P', 'PET 输入 (真实/合成)')
            
            # 特征提取
            cls.node('ResNet_M', 'ResNet 特征提取\n(MRI分支)')
            cls.node('ResNet_P', 'ResNet 特征提取\n(PET分支)')
            
            # CBAM
            cls.node('CBAM_M', 'CBAM 特征增强')
            cls.node('CBAM_P', 'CBAM 特征增强')
            
            # 交叉注意力 (关键部分)
            cls.node('CrossAttn', '基于局部感知的\n交叉注意力机制 (Cross-Attention)', width='3', style='filled', fillcolor='#BBDEFB')
            
            # 融合与分类
            cls.node('Fusion', '自适应门控融合\n(Adaptive Gating Fusion)', style='filled', fillcolor='#D1C4E9')
            cls.node('Classifier', '分类器')
            cls.node('Output', '诊断结果\n(AD / MCI / NC)', shape='doublecircle', style='filled', fillcolor='#FFECB3')

            # 连接分类网络内部
            cls.edge('Input_M', 'ResNet_M')
            cls.edge('Input_P', 'ResNet_P')
            cls.edge('ResNet_M', 'CBAM_M')
            cls.edge('ResNet_P', 'CBAM_P')
            
            # 汇入交叉注意力
            cls.edge('CBAM_M', 'CrossAttn')
            cls.edge('CBAM_P', 'CrossAttn')
            
            # 这里的trick是CrossAttn输出两路，但在图上为了简洁，通常画汇聚再分出，或者直接指向融合
            # 这里画两条线指向融合，代表增强后的特征
            cls.edge('CrossAttn', 'Fusion', label=' 增强特征融合')
            cls.edge('Fusion', 'Classifier')
            cls.edge('Classifier', 'Output')

    # === 3. 跨阶段连接 ===
    # 配对数据 -> I2SB训练 和 分类训练
    dot.edge('Data_Paired', 'Concat', style='dashed', label=' 训练I2SB')
    dot.edge('Data_Paired', 'MSE', style='dashed', label=' 监督信号')
    dot.edge('Data_Paired', 'Input_M', color='blue')
    dot.edge('Data_Paired', 'Input_P', color='blue')

    # 未配对数据 -> I2SB推理 -> 生成 -> 分类训练
    dot.edge('Data_Unpaired', 'Concat', label=' 推理')
    dot.edge('Gen_PET', 'Input_P', style='dashed', label=' 伪配对补充', color='red')
    dot.edge('Data_Unpaired', 'Input_M', color='red')

    # 渲染保存
    dot.render('AD_Diagnosis_Framework', view=True)
    print("流程图已生成：AD_Diagnosis_Framework.png")

if __name__ == '__main__':
    draw_thesis_diagram()