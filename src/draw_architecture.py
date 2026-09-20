# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path
from plot_style import configure_fonts, finish_fonts
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"savefig.dpi":300})
NAVY="#234E70"; BLUE="#3B82C4"; CYAN="#58B7C5"; ORANGE="#E58B3A"; RED="#C84B4B"; GREEN="#4D9078"; GRAY="#7A8793"
configure_fonts()

fig,ax=plt.subplots(figsize=(13.2,6.4))
ax.set_xlim(0,14); ax.set_ylim(0,6.6); ax.axis("off")

def box(x,y,w,h,text,color,fs=10.5,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=color,ec="none",alpha=0.96))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=tc,fontsize=fs,weight="bold",linespacing=1.45)
def arrow(s,e,color=GRAY,lw=1.8,style="-|>"):
    ax.add_patch(FancyArrowPatch(s,e,arrowstyle=style,mutation_scale=15,color=color,lw=lw,
                                 shrinkA=2,shrinkB=2))
# title
ax.text(7,6.35,"SafeFAME-TS v2 系统总体架构：双路候选、验证期资格门控与安全回退",
        ha="center",fontsize=15.5,weight="bold",color=NAVY)
# branch labels
ax.text(1.45,5.62,"数据层",ha="center",fontsize=10,color=GRAY,weight="bold")
ax.text(4.4,5.62,"表征与建模",ha="center",fontsize=10,color=GRAY,weight="bold")
ax.text(7.5,5.62,"候选构造",ha="center",fontsize=10,color=GRAY,weight="bold")

# Row positions
yu,yl=3.55,1.35; h=1.55
# 1 data
box(0.3,yu,2.3,h,"数值序列\n审计 · 稳定时间排序\n训练期标准化",NAVY)
box(0.3,yl,2.3,h,"外部事实文本\n严格时点 end_date<t\n排除 preds/pred",BLUE)
# 2 encode
box(3.1,yu,2.6,h,"数值专家池\nLast / 季节朴素 / AR-Ridge\nDLinear-M / PatchTST",CYAN)
box(3.1,yl,2.6,h,"冻结 MiniLM 句向量\nreport/search 时间衰减聚合\nPCA≤24 维 + 10 维质量",ORANGE)
# 3 build
box(6.2,yu,2.6,h,"校准段（验证前半）\n按 MSE 选定\n数值回退 b",GREEN)
box(6.2,yl,2.6,h,"残差候选（Last 锚 + Ridge）\n语义候选\n频率—语义候选",ORANGE)
# 4 gate
box(9.35,2.05,2.15,2.5,"验证决策段\n资格门控\n\n总体胜出\n且前后半均胜\n且置换 p≤0.025",RED,fs=10.5)
# 5 output
box(11.95,2.05,1.75,2.5,"安全路由\n\n证据充分\n→ 启用文本\n\n证据不足\n→ 数值回退",NAVY,fs=10.5)

# arrows data->encode
arrow((2.6,yu+h/2),(3.1,yu+h/2))
arrow((2.6,yl+h/2),(3.1,yl+h/2))
# encode->build
arrow((5.7,yu+h/2),(6.2,yu+h/2))
arrow((5.7,yl+h/2),(6.2,yl+h/2))
# build->gate
arrow((8.8,yu+h/2),(9.35,3.6))
arrow((8.8,yl+h/2),(9.35,3.0))
# gate->output
arrow((11.5,3.3),(11.95,3.3))

# frozen test band
box(0.3,0.15,13.4,0.95,"历史 v2：验证按起点分半，目标有重叠；审查后 v3 剔除跨界目标窗口\n99 次门控 ｜ 999 次敏感性 ｜ 5000 次块区间诊断；审查后沿用已查看测试集，不代表全局未见测试",
    "#F2F5F8",fs=10.5,tc=NAVY)
arrow((10.4,2.05),(10.4,1.12),color=GRAY,lw=1.4,style="-|>")
arrow((12.82,2.05),(12.82,1.12),color=GRAY,lw=1.4,style="-|>")

out=Path(__file__).resolve().parents[1]/"outputs/figures/fig_architecture.png"
finish_fonts(fig)
fig.savefig(out,bbox_inches="tight",facecolor="white")
print("saved",out)
