from pathlib import Path
from tokenizers import Tokenizer
import collections, hashlib, json, shutil, unicodedata

BASE=Path('/Users/zhangyunjian/Documents/Codex/2026-11-07/jia/outputs/data_pipeline')
PREV=BASE/'staging/round9/train'; NEXT=BASE/'staging/round10/train'
REV=BASE/'review'; REP=BASE/'reports'
queue=list(map(json.loads,(REV/'quarantined_records_round9.jsonl').open()))
by_id={x['record']['id']:x for x in queue}
assert len(queue)==51 and len(by_id)==51
legacy=list(map(json.loads,Path('/Users/zhangyunjian/Documents/Codex/2026-11-07/jia/work/legacy_article_en.jsonl').open()))
legacy_by_text={x['text']:x for x in legacy}
repairs={}
def add(n,text,lines='',url=None,note='checked against original page'):
    rid=('article-en-' if n.startswith('en-') else 'article-zh-')+n.removeprefix('en-')
    assert rid in by_id and rid not in repairs and text.strip()
    row=by_id[rid]['record']
    if url is None:
        title=row['source_title']; canon={'二分搜寻':'二分搜尋','简谐运动':'簡諧運動'}.get(title,title.replace(' ','_'))
        url='https://zh.wikipedia.org/wiki/'+canon
    repairs[rid]={'text':text,'source_url':url,'source_lines':lines,'note':note}

add('en-000487','High-frequency electromagnetic radiation involves heating the plasma by transferring energy from electromagnetic radiation to particles in the plasma.',url='https://www.energy.gov/science/doe-explainsburning-plasma',note='removed Resources and Related Terms navigation; historical exact-text provenance match')
add('en-000507','Fusion releases energy because the mass of its bound nucleus is less than the mass of its component protons and neutrons; the mass deficit is converted to energy through Einstein’s equation (E=mc²). A pickup truck filled with fusion fuel has the equivalent energy of 2 million metric tons of coal, or 10 million barrels of oil.',url='https://www.energy.gov/science/doe-explainsfusion-energy-science',note='restored superscript and removed Resources and Related Terms navigation; historical exact-text provenance match')
add('en-000534','The interaction between quarks and gluons is responsible for almost all the perceived mass of protons and neutrons and is therefore where we get our mass.',url='https://www.energy.gov/science/doe-explainsquarks-and-gluons',note='removed Resources and Related Terms navigation; historical exact-text provenance match')
add('en-000546','Thomson discovered the electron in 1897, and scientists at the Large Hadron Collider found the final piece of the puzzle, the Higgs boson, in 2012.',url='https://www.energy.gov/science/doe-explainsthe-standard-model-particle-physics',note='removed Resources and Related Terms navigation; historical exact-text provenance match')

add('000279','库仑定律指出，位于 r₁ 的电荷 q₁ 对位于 r₀ 的电荷 q₀ 施加的力为 F₀₁=(q₁q₀/4πε₀) r̂₀₁/|r₀₁|²=(q₁q₀/4πε₀) r₀₁/|r₀₁|³。其中，F₀₁ 是 q₁ 对 q₀ 造成的力，r̂₀₁ 是从 r₁ 指向 r₀ 的单位向量。','268-272')
add('000281','为了计算位于 r₀ 的任意电荷所受的库仑力，可将 F₀₁ 除以 q₀，得到源电荷 q₁ 产生的电场：E₁(r₀)=F₀₁/q₀=(q₁/4πε₀) r̂₀₁/|r₀₁|²=(q₁/4πε₀) r₀₁/|r₀₁|³。','274-275')
add('000284','若电荷 q₁,q₂,…,qₙ 静止于 R₁,R₂,…,Rₙ，且不存在电流，则由库仑定律与叠加原理得到 E(r)=ΣᵢEᵢ(r)=(1/4πε₀)Σᵢ qᵢ r̂ᵢ/|rᵢ|²=(1/4πε₀)Σᵢ qᵢ rᵢ/|rᵢ|³，其中求和范围为 i=1,…,n，r̂ᵢ 是从 Rᵢ 指向 r 的单位向量。','280-282')
add('000286','将空间中位置 r′ 处每个小体积 dv 内的电荷 ρ(r′)dv 视为点电荷。记 Δr=r−r′，则该微小电荷在 r 处造成的电场为 dE(r)=[ρ(r′)/(4πε₀)] Δr̂/|Δr|² dv=[ρ(r′)/(4πε₀)] Δr/|Δr|³ dv。其中 Δr̂ 是从 r′ 指向 r 的单位向量。','286-288',note='checked source formula; Δr notation makes source-to-field displacement explicit')
add('000287','把体积 V 中所有体积元的贡献相加，可得 E(r)=(1/4πε₀)∭_V ρ(r′)Δr/|Δr|³ dv，其中 Δr=r−r′。对于曲面 S 上的面电荷密度 σ(r′)，有 E(r)=(1/4πε₀)∬_S σ(r′)Δr/|Δr|³ da；对于曲线 L 上的线电荷密度 λ(r′)，有 E(r)=(1/4πε₀)∫_L λ(r′)Δr/|Δr|³ dℓ。静态系统中，磁场不随时间变化，电场为无旋场。','289-299',note='checked source integrals; Δr notation makes source-to-field displacement explicit')
add('000290','库仑定律给出电荷所受的力 F=q[Q r̂/(4πε₀|r|²)]=qE；牛顿万有引力定律给出 F=m[−GM r̂/|r|²]=mg，其中 r̂=r/|r|。电场 E 与重力场 g 及其相关势能函数有相似性。静电力与重力皆为中心力、保守力，并服从平方反比定律。两片平行导电板之间维持电压时，可近似产生均匀电场。','323-333')
add('000293','对特定系统求解后，可描述电磁场的整体行为。试探电荷在电磁场中受到的洛伦兹力为 F=qE+qv×B。电磁场的总能量密度为 u_EM=(ε/2)|E|²+[1/(2μ)]|B|²，其中 ε 是介质电容率，μ 是磁导率。由于 E 场与 B 场彼此耦合，电磁场及其能量的分解取决于参考系。','352-359')
add('000294','体积 V 中电磁场储存的总能量为 U_EM=(1/2)∫_V[ε|E|²+(1/μ)|B|²]dV。在有物质存在时，可用电位移场 D=ε₀E+P 描述系统，其中 P 是电极化，即电偶极矩的体积密度。E 与 P 分别独立定义，这一关系可用于定义 D。D 虽不如 E 与 P 直观，但可简化以自由电荷和自由电流表示的马克士威方程。','361-374')
add('000300','t_r 是延迟时间，即源发出对当前电场贡献的时刻；r_s(t) 是粒子的位置向量；n_s(r,t) 是从带电粒子指向场点的单位向量；β_s(t) 是粒子速度除以光速。延迟时间满足 t_r=t−|r−r_s(t_r)|/c。对于速度低于光速的带电粒子，在给定 t、r 和 r_s(t) 时，t_r 的解唯一。加速电荷的电磁辐射来自电场中依赖加速度的项。','412-413')
add('000301','马克士威方程还存在使用超前时间 t_a 的同形式解，满足 t_a=t+|r−r_s(t_a)|/c。这种解意味着某点的电场受未来粒子状态支配，通常被视为非物理解；费曼－惠勒吸收体理论也曾探讨它。上述方程尚未包含量子力学效应。处于静电平衡的导体表面，若某点面电荷密度为 σ，则紧贴表面的电场为 E=(σ/ε₀)x̂。','414-432')

# Further source-backed decisions are added below.
add('000179','根据牛顿万有引力定律，对于质量为 m₀、m 的两质点，位于 r 的质点 m 所受万有引力为 F(r)=−Gmm₀(r−r₀)/|r−r₀|³。其中 G 是万有引力常数，r₀、r 分别是两质点的位置矢量。','388-391')
add('000180','定义无穷远处为势能零点，则质点 m 在 r 处的引力势能为 E_G(r)=−Gmm₀/|r−r₀|。对于已知引力势分布 φ(r)，有 E_G(r)=mφ(r)。重力势能是引力势能在特殊情况下的简化形式。对于球对称分布的物体，其外部质点所受引力仍可由两质点公式描述，其中 m₀ 是球体总质量，r₀ 是球心位置。','391-405')
add('000182','在弹性限度内，弹簧弹力与长度变化量的关系为 F(x)=kx，其中 k 为弹簧弹性系数，x 为弹簧相对平衡位置的长度变化量。弹性势能为 E_p(x)=½kx²，通常以弹簧原长 x=0 为势能零点。在静电学中，两静止点电荷 q、q₀ 之间的静电力为 F(r)=[qq₀/(4πε₀)](r−r₀)/|r−r₀|³，其中 ε₀ 是电常数。','427-447',note='source formula F(x)=kx preserved; source itself omits restoring-force sign convention')
add('000359','对于双变数实函数 g，其元素可写为 ((x,y),z)，其中 x,y,z∈ℝ。令 Dₓ={x | ∃y∃z，g(x,y)=z}、Dᵧ={y | ∃x∃z，g(x,y)=z}。以 x、y 轴分别表示 Dₓ、Dᵧ 的变化范围，以 z 轴表示 g 的值域，即可在三维直角坐标中绘出函数图形。平面图形可用竖直线判别法检验是否为单变数实函数的图形。','530-544',note='restored formulas; corrected source page apparent variable typo in D_y set')
add('000360','平行于 y 轴、经过 (c,0) 的直线可表示为 {p∈ℝ² | ∃y∈ℝ，p=(c,y)}。它与实函数 f 的图形的交集为 {p∈ℝ² | ∃y∈ℝ，p=(c,y) 且 y=f(c)}。交点最多只有一个，否则会违背每个定义域元素只对应一个函数值的定义。','544-551')
add('000362','若 f:X→Y 且 X′⊂X，则 f 在 X′ 上的限制定义为 f|_{X′}={(x,y) | x∈X′ 且 y=f(x)}=f∩(X′×Y)，它仍是函数。反之，若 g:X→Z、X⊆Y、f:Y→Z 且 f|_X=g，则称 f 为 g 的扩张。解析函数的扩张称为解析延拓。','580-585')
add('000363','设 f,g:X→R，且 (R,+,×) 为环。可逐点定义函数和 (f+g)(x)=f(x)+g(x)，以及函数积 (fg)(x)=f(x)g(x)。对 r∈R，常数函数 r_R 满足 r_R(x)=r；数乘函数满足 (r·f)(x)=r f(x)。','590-596')
add('000465','以平方根函数 f(x)=√x 为例，它在区间 [0,1] 上图形下方的面积记为 S=∫₀¹√x dx。其中 dx 表示积分变量，0 和 1 分别是积分下界与上界，√x 是被积函数。为近似求取面积，可将 [0,1] 等分为 [0,0.2)、[0.2,0.4)、[0.4,0.6)、[0.6,0.8)、[0.8,1]，并在每段上用矩形覆盖图形。','559-565')
add('000474','设 S 为函数 f 在闭区间 [a,b] 上的黎曼积分。当且仅当对任意 ε>0，存在 δ>0，使任意取样分割 a=x₀<⋯<xₙ=b，tᵢ∈[xᵢ,xᵢ₊₁]，只要最大子区间长度 λ≤δ，就有 |Σᵢ₌₀ⁿ⁻¹ f(tᵢ)(xᵢ₊₁−xᵢ)−S|<ε。也就是说，所有足够细的取样分割得到的黎曼和都趋向同一个值 S。','594-602')
add('000484','若 f 在区间上黎曼可积，则积分区间具有可加性：∫ₐᶜf(x)dx=∫ₐᵇf(x)dx+∫ᵦᶜf(x)dx。若 f 在两个不相交的可测集 I、J 上勒贝格可积，则 ∫_{I∪J}f dμ=∫_I f dμ+∫_J f dμ。积分还具有绝对连续性：对任意 ε>0，存在 δ>0，使 μ(A)<δ 时 ∫_A|f|dμ<ε。','659-663')
add('000489','赫尔德不等式的积分形式：若 p,q>0 且 1/p+1/q=1，则在各积分有意义时，|∫f(x)g(x)dx|≤(∫|f(x)|ᵖdx)¹ᐟᵖ(∫|g(x)|ᑫdx)¹ᐟᑫ。p=q=2 时得到柯西不等式。','687-689')
add('000490','闵可夫斯基不等式的积分形式：若 p≥1，则在各积分有意义时，(∫|f(x)+g(x)|ᵖdx)¹ᐟᵖ≤(∫|f(x)|ᵖdx)¹ᐟᵖ+(∫|g(x)|ᵖdx)¹ᐟᵖ。勒贝格可积函数的相关不等式可用于构建 Lᵖ 空间。对于可积函数 f，还有 |∫_I f|≤∫_I|f|。','691-694')
add('000534','若两个向量长度和方向均相同，它们是相等向量；若 a−b=0，则 b 是 a 的相等向量。方向相同而大小可以不同的向量互为方向向量。有向线段在方向和长度之外还指定起点与终点；从 A 到 B 的线段长度记为 |AB⃗|，并有有向数量 AB=−BA。','378-400')
add('000536','对于 n 维欧几里得空间中的向量 v=(v₁,v₂,…,vₙ)，其模长为 |v|=‖v‖=√(v₁²+v₂²+⋯+vₙ²)。三维向量 a=(x,y,z) 的模长为 ‖a‖=√(x²+y²+z²)。两个向量的夹角是它们在图示中两箭头之间的角。','407-416')
add('000545','标量 −1 乘以任意向量得到其反向量，0 乘以任意向量得到零向量。两个向量 a、b 的内积是标量，几何表达式为 a·b=|a||b|cosθ，其中 θ 为夹角。力做功的一个例子是 W=F·s。外积的结果是垂直于两个输入向量所在平面的向量，方向由右手定则确定。','475-495')
add('000547','设 a=aₓi+aᵧj+a_zk，b=bₓi+bᵧj+b_zk，则 a×b 可写为以 i、j、k 为第一行，aₓ、aᵧ、a_z 为第二行，bₓ、bᵧ、b_z 为第三行的三阶行列式。三个向量的标量三重积满足 a·(b×c)=b·(c×a)=c·(a×b)。若 OA⃗、OB⃗ 是平面上两个不平行的非零向量，则任意 OP⃗ 都可唯一写为 OP⃗=xOA⃗+yOB⃗。','496-514')
add('000549','在平面直角坐标系中，设 A=(x₁,y₁)、B=(x₂,y₂)，P 在两点之间，且 |AP⃗|:|PB⃗|=n，则 P=((x₁+nx₂)/(1+n),(y₁+ny₂)/(1+n))。当 n=1 时，P 是中点，坐标为 ((x₁+x₂)/2,(y₁+y₂)/2)。','524-532')
add('000541','两个向量 a、b 起点重合时，a−b 可表示为从 b 的终点指向 a 的终点的向量。在三维基向量 e₁=(1,0,0)、e₂=(0,1,0)、e₃=(0,0,1) 下，若 a=Σaᵢeᵢ、b=Σbᵢeᵢ，则 a+b=(a₁+b₁)e₁+(a₂+b₂)e₂+(a₃+b₃)e₃。模长满足 |a|+|b|≥|a+b|≥|a|−|b|。','437-452')
add('000139','角动量算符 L² 与 L_z 的共同特征态 |l,m⟩ 满足 L²|l,m⟩=ℏ²l(l+1)|l,m⟩，L_z|l,m⟩=ℏm|l,m⟩。在角坐标中，⟨θ,φ|l,m⟩=Y_{l,m}(θ,φ)，即球谐函数。其中 l 是非负整数，m 是满足 −l≤m≤l 的整数。经典力学中角动量可连续取值；角动量量子化则给出离散值。','231-247')
add('000370','实数数列 {aₙ} 收敛于 L，指对任意 ε>0，都存在正整数 N，使得所有 n≥N 时有 |aₙ−L|<ε。等差数列从第二项起相邻两项之差固定，例如 1,3,5,7,…，其公差 d=2。若首项为 a₁，通项公式为 aₙ=a₁+(n−1)d。','208-229')
add('000447','二阶导数的导数称为三阶导数，记为 f‴(x)、y‴、d³y/dx³ 或 d³f(x)/dx³。三阶导数的导数称为四阶导数，记为 f⁽⁴⁾(x)、y⁽⁴⁾、d⁴y/dx⁴ 或 d⁴f(x)/dx⁴。一般地，f(x) 的 n−1 阶导数的导数称为 n 阶导数，记为 f⁽ⁿ⁾(x)、y⁽ⁿ⁾、dⁿy/dxⁿ 或 dⁿf(x)/dxⁿ。','816-823')
add('000561','以列举元素的方式表示一个 3×2 矩阵，可写成 {((1,1),a),((1,2),b),((2,1),c),((2,2),d),((3,1),e),((3,2),f)}，通常排成三行两列：第一行为 a、b，第二行为 c、d，第三行为 e、f。一般的 m×n 矩阵写作 A=[aᵢⱼ]_{m×n}，i 表示行位置，j 表示列位置。','310-318')
add('000621','对于实矩阵 A，若 AᵀAx=0，则 0=xᵀAᵀAx=‖Ax‖²，因此 Ax=0；所以 AᵀA 与 A 的零空间相同，并有 rank(AᵀA)=rank(A)。对复矩阵 A，以 A* 表示共轭转置，则 rank(A)=rank(Ā)=rank(Aᵀ)=rank(A*)=rank(A*A)。把 m 个 n 维列向量排成 n×m 矩阵 A，其秩就是原向量组的秩。','287-297')
add('000418','按函数极限定义，可直观地令 h=x−a 并使 h→0，写出 f′(a)=lim_{h→0}[f(a+h)−f(a)]/h。当 f 只在定义域 D_f 的部分点有定义时，这一写法应理解为原导数极限的简记。若存在 r>0，使 f 在 (a−r,a+r) 上有定义，可令 F 的定义域为 {h∈ℝ | h≠0 且 |h|<r}，并定义 F(h)=[f(a+h)−f(a)]/h；此时 lim_{h→0}F(h)=f′(a)。','570-583')
add('000642','幂法从一个随机向量 v 出发，依次计算单位向量 Av/‖Av‖、A²v/‖A²v‖、A³v/‖A³v‖，等等。在适当条件下，序列趋向于模最大的特征值对应的特征向量。特征值 λ 的代数重数是它作为特征多项式根的重数；若 r 是根，即因子 (λ−r) 在特征多项式的分解中出现的次数。按代数重数计，n×n 矩阵有 n 个特征值。','366-377')
add('000650','无穷维空间中可用谱推广特征值的概念。算子 T 的谱是使 (T−λ)⁻¹ 不存在有界逆的标量 λ 的集合。特征值一定属于谱，但谱值未必是特征值。例如希尔伯特空间 ℓ²(ℤ) 由平方和 Σ_{k∈ℤ}|aₖ|² 收敛的双向序列构成；其上的双向平移算子没有特征向量，却有谱值。','461-467')
add('000652','在平方可积函数组成的希尔伯特空间中，可选取基并把波函数 Ψ_E 与哈密顿算子 H 分别表示成向量和矩阵，从而用矩阵形式表达薛定谔方程。狄拉克标记区分态向量 |Ψ_E⟩ 与位置空间波函数 Ψ_E(x)。方程写为 H|Ψ_E⟩=E|Ψ_E⟩，其中 |Ψ_E⟩ 是 H 的本征态，E 是对应的能量本征值。','479-487')
add('000656','惯性张量 I 是实值三维对称矩阵，其对角元素 I_xx、I_yy、I_zz 是刚体绕 x、y、z 轴的转动惯量；非对角元素 I_αβ（α≠β）是相应的惯量积。求主轴可解特征方程 Iω=λω，等价于三阶行列式 det(I−λI₃)=0，其中矩阵三行依次为 (I_xx−λ,I_xy,I_xz)、(I_yx,I_yy−λ,I_yz)、(I_zx,I_zy,I_zz−λ)。','519-534')
add('000673','在三个抽屉中按掷骰子结果选一个，再随机取一个球，取到白球的全概率为 P(白)=P(白|抽屉1)P(抽屉1)+P(白|抽屉2)P(抽屉2)+P(白|抽屉3)P(抽屉3)。若三个抽屉的白球比例依次为 14/20、2/10、3/10，选中概率依次为 3/6、2/6、1/6，则 P(白)=28/60≈0.4667。贝叶斯定理用于联系 P(A|B) 与 P(B|A)。','449-468')
add('000674','由乘法法则 P(A∩B)=P(A)P(B|A)=P(B)P(A|B)，得贝叶斯公式 P(A|B)=P(B|A)P(A)/P(B)。例如某别墅过去 20 年被盗 2 次，狗平均每周夜间叫 3 次，盗贼入侵时狗叫的估计概率为 0.9；可用该公式估计听到狗叫时发生入侵的概率。','468-471')
add('000675','设 A 为狗在夜间叫，B 为盗贼入侵。按原例的估算，P(A)=3/7，P(B)=2/(20×365.25)=2/7305，P(A|B)=0.9。因此 P(B|A)=0.9×(2/7305)×(7/3)≈0.00057495。另一个例子：容器 A 有 7 个红球、3 个白球，容器 B 有 1 个红球、9 个白球，从两个容器任选一个球且已知是红球，求它来自 A 的概率。','471-474')
add('000676','在两个容器的例子中，设 A 为球来自容器 A，B 为抽到红球。则 P(B)=8/20，P(A)=1/2，P(B|A)=7/10，由贝叶斯公式得 P(A|B)=(7/10)×(1/2)×(20/8)=7/8。概率论在物理、化学、生物、医学、心理学、社会科学和工程学中有广泛应用，也是数理统计的基础。','474-487')
add('000738','Immerman–Szelepcsényi 定理表明：若 f∈Ω(log n)，则 NSPACE(f(n)) 对取反封闭；也就是说，如果语言 A∈NSPACE(f(n))，其补集 Aᶜ={x | x∉A} 也属于 NSPACE(f(n))。对数空间复杂度类 L（或 LOGSPACE）是确定性图灵机只用 O(log n) 工作空间即可判定的语言类。','158-163')
add('000784','关联矩阵是以顶点为行、边为列的二维矩阵，元素标识顶点与边的关系。记 |V| 为顶点数、|E| 为边数。邻接矩阵常用于较稠密的图，即 |E| 接近 |V|²，或需要频繁查询两节点是否邻接的场景。其他图结构包括链式前向星、十字链表和邻接多重表。并行图计算要考虑数据规模、分布与访问开销；图的表示和存储方式会影响通信成本及可扩展性。','191-212')
add('000850','设 a∈X，f:X→X。若函数 F,G:ℕ→X 都满足 F(0)=G(0)=a，且 F(n+1)=f(F(n))、G(n+1)=f(G(n))，则可用数学归纳法证明 F(n)=G(n) 对所有 n∈ℕ 成立：n=0 时成立；若 F(k)=G(k)，则 F(k+1)=f(F(k))=f(G(k))=G(k+1)。','288-310')
add('001040','设第 l 层输出 yᶩᵢ=fᶩ(Σⱼ yᶩ⁻¹ⱼWᶩⱼᵢ)，损失为 g。链式法则给出该层权重梯度：∂g/∂Wᶩ_cd=(∂g/∂yᶩ_d) fᶩ′(Σₐ yᶩ⁻¹ₐWᶩ_ad)yᶩ⁻¹_c；传向前一层的梯度为 ∂g/∂yᶩ⁻¹_c=Σᵢ[(∂g/∂yᶩ_i) fᶩ′(Σₐ yᶩ⁻¹ₐWᶩ_ai)Wᶩ_ci]。从输出层向前逐层计算梯度，称为反向传播。','379-399',note='source-backed compact equivalent of original nested g^l derivation')
add('001066','L₀ 正则项不连续、非凸且不可微，因此原文转而讨论其最紧凸放松 L₁ 范数。L₁ 正则项可写为 Ω(F(x;ω))=γ₁‖ω‖₁/n，其中 γ₁>0。它对每个参数 ωᵢ 在结构风险上施加大小为 γ₁|ωᵢ|/n 的惩罚，以促进参数稀疏化；L₁ 正则项也称 LASSO 正则项。过拟合时，模型曲线可能在噪声点附近剧烈弯折。','346-356')

NEXT.mkdir(parents=True,exist_ok=True)
old={}
for name in ('natural_zh.jsonl','natural_en.jsonl','article_zh.jsonl','article_en.jsonl'):
    old[name]=list(map(json.loads,(PREV/name).open()))
groups={x['source_title']:x['group_id'] for x in old['article_zh.jsonl']}
assert all(x['group_id']==groups[x['source_title']] for x in old['article_zh.jsonl'])
log=[]
for rid,decision in sorted(repairs.items()):
    item=by_id[rid]; row=item['record'].copy(); before=row['text']; row['text']=decision['text']
    if rid.startswith('article-zh-'):
        row['group_id']=groups[row['source_title']]
        name='article_zh.jsonl'
    else:
        assert before in legacy_by_text
        hist=legacy_by_text[before]
        for field in ('source','source_url','source_title','chunk_index'):
            row[field]=hist[field]
        row['legacy_id']=hist['id']
        row['group_id']='article-url-'+hashlib.sha256(row['source_url'].encode()).hexdigest()[:16]
        name='article_en.jsonl'
    assert all(x['id']!=rid for x in old[name])
    old[name].append(row)
    log.append({'record_id':rid,'decision':'source_verified_repair_and_restore','source_url':decision['source_url'],'source_lines':decision['source_lines'],'source_checked_on':'2026-11-07','note':decision['note'],'prior_flags':item['flags'],'old_text_sha256':hashlib.sha256(before.encode()).hexdigest(),'new_text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'old_text':before,'new_text':row['text']})
old['article_zh.jsonl'].sort(key=lambda x:int(x['id'].rsplit('-',1)[-1]))
old['article_en.jsonl'].sort(key=lambda x:int(x['id'].rsplit('-',1)[-1]))
tok=Tokenizer.from_file('/Users/zhangyunjian/felixspace/rat/gpt2/tokenizer/tokenizer.json');tok.no_truncation()
stats={}
for name,rows in old.items():
    p=NEXT/name;p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    stats[name]={'records':len(rows),'tokens':sum(len(tok.encode(x['text'],add_special_tokens=False).ids) for x in rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
remaining=[x for x in queue if x['record']['id'] not in repairs]
(REV/'quarantined_records_round10.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in remaining))
(REP/'source_repair_decisions_round10.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in log))
(REP/'staging_round10_manifest.json').write_text(json.dumps({'input_staging':str(PREV),'output_staging':str(NEXT),'repaired_and_restored':len(repairs),'risk_still_quarantined':len(remaining),'files':stats,'status':'provisional; external rights and random sample review outstanding'},ensure_ascii=False,indent=2)+'\n')
ids=[x['id'] for rows in old.values() for x in rows];assert len(ids)==len(set(ids))
norm=[' '.join(unicodedata.normalize('NFC',x['text']).split()) for rows in old.values() for x in rows]
assert all(norm) and len(norm)==len(set(norm))
print(json.dumps({'repaired':len(repairs),'remaining':len(remaining),'records':len(ids),'tokens':sum(x['tokens'] for x in stats.values())},ensure_ascii=False))
