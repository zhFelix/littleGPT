from pathlib import Path
import collections, hashlib, json, re, shutil
from tokenizers import Tokenizer

BASE = Path('/Users/zhangyunjian/Documents/Codex/2026-11-07/jia/outputs/data_pipeline')
PREV = BASE / 'staging/round8/train'
NEXT = BASE / 'staging/round9/train'
REVIEW = BASE / 'review'
REPORT = BASE / 'reports'

repairs = {}
def set_text(n, text, lines):
    repairs[f'article-zh-{n}'] = (text, lines)

rows = [json.loads(line) for line in (REVIEW / 'quarantined_records.jsonl').open()]
source = {x['record']['id']: x for x in rows}
def old(n): return source[f'article-zh-{n}']['record']['text']
def replace(n, pairs, lines):
    text = old(n)
    for before, after in pairs:
        assert text.count(before) >= 1, (n, before, text.count(before))
        text = text.replace(before, after)
    set_text(n, text, lines)

replace('000081', [
    ('位纯量场U(r) )\n', '位纯量场 U(r) '),
    ('相反：F−∇U. =- U.', '相反：F = −∇U。'),
], '495-506')
replace('000168', [
    ('洛伦兹力Fq(E+vB\n', '洛伦兹力 F = q(E + v×B)。'),
    ('定义电势与矢势A', '定义电势 φ 与矢势 A'),
    ('函数Uq(−A⋅v) )\n', '函数 U = q(φ − A·v)。'),
], '273-286')
replace('000224', [
    ('xAcos(t+) )', 'x = A cos(ωt + φ)'),
    ('v-Asin(t+) )', 'v = −Aω sin(ωt + φ)'),
    ('a-A2cos(t+) ', 'a = −Aω² cos(ωt + φ)。'),
    ('aFm-kxm-Acos(t+)k\nm m= -kxm= -A km两式联立得km m\n。', 'a = F/m = −kx/m = −Ak cos(ωt + φ)/m。两式联立得 ω = √(k/m)。'),
], '237-245')
set_text('000259', '反射角等于入射角：∠θᵢ=∠θᵣ。光从不同密度的介质穿过时发生的偏折现象称为折射。不同介质可以出现不同的折射角，折射率可写作 n=sin(∠θ₁)/sin(∠θ₂)，并遵从斯涅尔定律：n₁sin(∠θ₁)=n₂sin(∠θ₂)。光速在不同介质中也会改变：v₂=v₁sin(∠θ₂)/sin(∠θ₁)。当 ∠θ₂=90° 时，折射光沿界面运行，∠θ₁ 称为临界角 ∠θ_c；当 ∠θ₁>∠θ_c 时，入射光完全反射回原介质，称为全内反射。', '444-465')
replace('000231', [
    ('频率与波矢量k之间的关系：(k) )\n。', '频率ω与波矢量 k 之间的关系：ω = ω(𝐤)。'),
    ('即k=2/ 。', '即 k = 2π/λ。'),
    ('波的相速度vp/与群速度vgd/d\n。', '波的相速度 vₚ = ω/k 与群速度 𝐯_g = dω/d𝐤。'),
], '308-316')
set_text('000285', 'rᵢ 是从点 Rᵢ 到点 r 的位移向量。叠加原理可用来计算电荷密度分布 ρ(r) 所造成的电场。', '282-286')
replace('000341', [
    ('(x，x2) )\n', '(x,x²)\n'),
    ('唯一值x2 的对应关系', '唯一值 x² 的对应关系'),
], '306-314')
set_text('000354', '因为单元对被规定成 ⟨x⟩:=x，前面的函数 h 就可以简记为 h(x)=x+1（x∈ℝ）。如果定义域可以从上下文推断出来，函数记号也可以更简略地写为 f(x₁,…,xₙ)=T。例如，函数 h 可以进一步简记为 h(x)=x+1。这个记号是莱昂哈德·欧拉于 1734 年首次采用的；当时还没有清楚地区分函数、项与幂级数。', '452-466')
set_text('000361', '像可以指两种不同的概念。第一种是形如 f(x) 的项，直观上代表按照函数 f 的对应规则，x 对应到的值。第二种指集合 A 在函数 f 下定义的集合 f(A)：f(A):={y | 存在 x∈A，使 y=f(x)}。函数 f 的值域就是定义域 D_f 的像 f(D_f)。例如，集合 {2,3} 在 f 下的像是 f({2,3})={c,d}，而 f 的值域是 {c,d}。', '556-562')
set_text('000400', '具体来说，设 f 是一个函数，并在定义域内取点 a，则 (a,f(a)) 是函数图像上的一个点。假设 h 接近于 0，则 a+h 接近于 a，点 (a+h,f(a+h)) 也接近 (a,f(a))。这两点之间的斜率为 m=[f(a+h)−f(a)]/[(a+h)−a]=[f(a+h)−f(a)]/h。这个表达式称为差商。通过曲线上两点的直线称为割线，因此 m 是这两点间割线的斜率。', '682-685')
set_text('000415', '导数（英语：derivative）是微积分学中的一个概念。函数在某一点的导数是指这个函数在这一点附近的变化率，即函数在该点的切线斜率。导数的本质是通过极限的概念对函数进行局部线性逼近。当函数 f 的自变量在点 x₀ 产生增量 h 时，如果函数输出值的增量与 h 的比值在 h→0 时的极限存在，该极限就是 f 在 x₀ 处的导数，记作 f′(x₀)、(df/dx)(x₀) 或 (df/dx)|_{x=x₀}。例如在运动学中，物体的位移对时间的导数就是物体的瞬时速度。不是所有的函数都有导数，一个函数也不一定在所有的点上都有导数。', '550-551')
set_text('000420', '若曲线是函数 y=f(x) 的图像，割线 PP₀ 的斜率为 tanφ=Δy/Δx=[f(x₀+Δx)−f(x₀)]/Δx。当点 P₀ 处的切线 P₀T，即割线 PP₀ 的极限位置存在时，Δx→0，φ→α，则切线斜率为 tanα=lim_{Δx→0}[f(x₀+Δx)−f(x₀)]/Δx。上式与导数的定义相同，即 f′(x₀)=tanα。因此，导数的几何意义是曲线 y=f(x) 在 P₀(x₀,f(x₀)) 处切线的斜率。', '587-593')
set_text('000440', '对于满足 f′(x₀)=0 的点 x₀，如果存在 δ>0，使 f′ 在区间 (x₀−δ,x₀] 上都大于等于零，而在 [x₀,x₀+δ) 上都小于等于零，那么 x₀ 是极大值点；反之为极小值点。如果 f″(x₀)=0，且 f″(x) 在 x₀ 两侧变号，则 x₀ 是拐点。如果 f″(x₀) 存在，且已知 f′(x₀)=0，那么也可以用二阶导数的符号判断极值：f″(x₀)>0 时是极小值点，f″(x₀)<0 时是极大值点。', '723-724')
set_text('000449', '当函数 y 的取值是 ℝⁿ 中的向量时，仍可对它求导。这时函数值为 y=(y₁(x),y₂(x),…,yₙ(x))，其中每个 yᵢ(x)（1≤i≤n）都是实数值函数。例如二维或三维空间中的参数方程。对 y=f(x) 求导，实际上就是对每个分量函数 yᵢ(x) 求导。', '843-847')
set_text('000472', '对于在闭区间 [a,b] 上有定义的实值函数 f，关于取样分割 x₀,…,xₙ 和 t₀,…,tₙ₋₁ 的黎曼和定义为 Σ_{i=0}^{n−1} f(tᵢ)(xᵢ₊₁−xᵢ)。和式中每一项都是子区间长度 xᵢ₊₁−xᵢ 与该区间内取样点 tᵢ 处函数值 f(tᵢ) 的乘积。直观地看，这是以标记点到 X 轴的距离为高、以子区间长度为底的矩形面积。最简单的取样分割方法是将区间均匀分割，再按相同准则选取每个子区间的标记点。', '597-600')
set_text('000486', '如果两个函数几乎处处相同，那么它们的积分相同。如果对可测集合族中的任意集合 A，可积函数 f 在 A 上的积分总等于（或大于等于）可积函数 g 的积分，那么 f 几乎处处等于（或大于等于）g。如果 f 在区域 I 上可积，M 和 m 分别是 f 在 I 上的最大值和最小值，那么 mL(I)≤∫_I f≤ML(I)。其中 L(I) 在黎曼积分中表示区间长度，在勒贝格积分中表示集合的测度。积分的绝对连续性表明：如果函数在某区域上可积，那么积分区域逐渐接近全区域时，积分值也接近全区域上的积分值。', '667-676')
set_text('000508', '例如，若已知 y(π)=2，则可推出 C=1，从而得到 y=−cos x+1。对于一阶线性常微分方程 y′+p(x)y+q(x)=0，常用常数变易法：设 y=C(x)e^{−∫p(x)dx}，将其代回原方程，即可求出 C(x)。对于二阶常系数线性常微分方程 y″+py′+qy=f(x)，先求特征方程 r²+pr+q=0 的根，得到齐次方程的解 y_h=c₁y₁+c₂y₂。若 r₁=r₂=r，则 y_h=(C₁+C₂x)e^{rx}；若 r₁≠r₂，则 y_h=C₁e^{r₁x}+C₂e^{r₂x}；若为共轭复数根 α±βi，则 y_h=e^{αx}[C₁cos(βx)+C₂sin(βx)]。再求特解 y_p，最终得到 y=y_h+y_p。微分方程的约束条件，是指其解需要满足的条件。', '465-510')
replace('000551', [
    ('f(x)4\nx', 'f(x) = 4x'),
    ('m1 的矩阵', 'm×1 的矩阵'),
    ('f(v) )', 'f(v)'),
    ('可以表示成A\n的形式', '可以表示成 Av 的形式'),
], '259-261')
set_text('000566', '矩阵乘法与转置满足 (AB)ᵀ=BᵀAᵀ。矩阵乘法不满足交换律。一般来说，矩阵 A、B 的乘积 AB 存在，但 BA 不一定存在；即使存在，大多数时候 AB≠BA。这一特性使矩阵代数与常见数域以及多项式环、整数环不同。给定一个 n 维方块矩阵 A，与 A 交换的所有方块矩阵构成一个环，称为 A 的交换子环；这些矩阵也构成 M(n,ℝ) 的一个子空间，称为 A 的可交换空间。', '371-377')
set_text('000570', '矩阵 A_f 第 i 行第 j 列上的元素，是正则基向量 e_j=(0,…,0,1,0,…,0)ᵀ（第 j 个元素是 1，其余元素是 0）在 f 映射后的向量 f(e_j) 的第 i 个元素。也就是说，从 ℝⁿ 射到 ℝᵐ 的线性变换构成的向量空间 L(ℝⁿ,ℝᵐ) 与 m×n 实矩阵空间 M(m,n,ℝ) 一一对应，映射为 f↦A_f。以下是一些典型的二维实平面线性变换对平面向量造成的效果及其对应的二维矩阵。其中，仅有伸缩（或压缩）变换会影响变换后的图形面积。', '398-400')
set_text('000574', '方块矩阵 A 的行列式是一个将它映射到标量的函数，记作 det(A) 或 |A|。一个方阵的行列式等于 0，当且仅当该方阵不可逆。系数为实数时，二维（三维）方阵 A 的行列式绝对值，表示单位面积（体积）的图形经过 A 对应的线性变换后得到的面积（体积）；正负号表示是否改变空间定向。2×2 矩阵的行列式为 |a b; c d|=ad−bc。更高维矩阵的行列式可以使用莱布尼兹公式，或通过拉普拉斯展开递推计算。', '430-435')
set_text('000575', '两个矩阵相乘，乘积的行列式等于各自行列式的乘积：det(AB)=det(A)·det(B)。将矩阵的一行或一列乘以某个系数后加到另一行或列上，不改变行列式；交换两行或两列会使行列式变号。利用这些操作可以将矩阵变为三角矩阵，其行列式是主对角线元素的乘积。行列式也可用于计算线性方程组的解。n×n 方块矩阵 A 的特征值 λ 和对应的非零特征向量 v 满足 Av=λv。特征值和特征向量有助于研究线性变换。', '436-445')
set_text('000600', '设有二阶拟线性偏微分方程 (E)：Σ_{1≤i,j≤n} aᵢⱼ ∂²f/(∂xᵢ∂xⱼ)+Σ_{i=1}^{n} bᵢ ∂f/∂xᵢ+cf=g，并假设 aᵢⱼ=aⱼᵢ。记矩阵 A=[aᵢⱼ]_{1≤i,j≤n}。如果 A 是正定或负定矩阵，则称方程 (E) 为椭圆型偏微分方程；如果 A 不可逆，称为抛物型偏微分方程；如果 A 可逆且恰有 n−1 个特征值同号，称为双曲型偏微分方程。', '591-593')
set_text('000601', '其它情况下，方程 (E) 也称为超双曲型偏微分方程。用数值方法解偏微分方程时也会用到矩阵。有限元方法的基本思想，是用一系列简单函数的线性组合来逼近偏微分方程的精确解。这些函数通常是将求解区域分成小块后，仅在某一小块上非零的分段线性函数。选定网格和函数后，可以求解刚度矩阵方程得到近似解；在一定条件下，近似解随着网格趋于精细而弱收敛到精确解。概率论中常用随机矩阵定义有限概率空间中的马尔可夫链。', '593-598')
replace('000705', [
    ('O(logn)', 'O(log n)'), ('O(nlogn)', 'O(n log n)'),
    ('O(n2) )', 'O(n²)'), ('O(n3) )', 'O(n³)'),
    ('O(nk) )', 'O(nᵏ)'), ('O(2n) )', 'O(2ⁿ)'),
], '350-360')
set_text('000730', '但是输入规模往往要大到不切实际，运行时间才会真正超过低阶多项式。准多项式时间算法慢于多项式时间算法，但通常快于指数时间算法。对于固定常数 c>0，其最坏情况运行时间可写作 2^{O((log n)^c)}；当 c=1 时得到多项式时间。术语“次指数时间”用于描述增长速度可能超过任何多项式、但明显小于指数的运行时间。次指数时间没有统一的严格定义，常用定义有两种。', '300-317')
set_text('000731', '一种定义是：对于每个 ε>0，如果都存在一个能在 O(2^{n^ε}) 时间内解决问题的算法，则称该问题属于次指数时间。此类问题的集合记为 SUBEXP=⋂_{ε>0}DTIME(2^{n^ε})。另一种定义将次指数时间写作 2^{o(n)}，允许的运行时间范围更广。普通数域筛选法是整数因式分解的一种算法，其运行时间约为 2^{Õ(n^{1/3})}；图同构问题的某种算法运行时间为 2^{O(√(n log n))}。', '313-323')
replace('000734', [
    ('O(nlogn)', 'O(n log n)'),
    ('O(n) )', 'O(n^α)'), ('O(2n) )\n', 'O(2ⁿ)\n'),
], '132-140')
replace('000735', [
    ('DTIME(f(n)、NTIME(f(n)', 'DTIME(f(n))、NTIME(f(n))'),
    ('DSPACE(f(n)和NSPACE(f(n)', 'DSPACE(f(n)) 和 NSPACE(f(n))'),
    ('O(f(n)\n) 大小', 'O(f(n)) 大小'),
], '139-141')
replace('000759', [('O(n) )\n的复杂度', 'O(√n) 的复杂度')], '250-252')
replace('000799', [
    ('大小(\nn )', '大小 n'),
    ('O(nlogn\n)', 'O(n log n)'),
    ('O(n2) )', 'O(n²)'),
    ('存在 (nlogn\n) 的下界', '存在 Ω(n log n) 的下界'),
], '191-194')
replace('000802', [
    ('O(n2) )', 'O(n²)'),
    ('O(nlogn)', 'O(n log n)'),
    ('O(nlog2n) n)', 'O(n log² n)'),
    ('二叉排序树排序（binary tree sort）—O(n log n) 期望时间；', '二叉排序树排序（binary tree sort）—O(n log n) 期望时间，O(n²) 最坏时间；需要 O(n) 额外空间。'),
    ('额外空间原地归并排序', '额外空间。\n原地归并排序'),
], '216-227')
replace('000803', [
    ('需要O(n) 额外空间\n鸽巢排序', '鸽巢排序'),
    ('O(n2) )', 'O(n²)'),
    ('O(nlogn)', 'O(n log n)'),
    ('(1+)n', '(1+ε)n'),
    ('O(nlogn\n)', 'O(n log n)'),
    ('图书馆排序（library sort）—O(n log n) 期望时间；', '图书馆排序（library sort）—O(n log n) 期望时间，O(n²) 最坏时间；'),
    ('块排序（block sort）—O(n log n) Tim排序', '块排序（block sort）—O(n log n)。\nTim排序'),
    ('Tim排序（Timsort）—O(n log n) 平均、最坏时间；', 'Tim排序（Timsort）—O(n log n) 平均、最坏时间；O(n) 最优时间，需要 O(n) 额外空间。'),
], '228-233')
replace('000804', [
    ('是目前已知最快的排序算法，在Python、Swift、Rust等语言的内置排序功能中被用作默认算法\n', ''),
    ('O(n2) )', 'O(n²)'),
    ('O(nlog2n) n)', 'O(n log² n)'),
    ('O(nlogn\n)', 'O(n log n)'),
    ('O(nlogn+k\n)', 'O(n log n + k)'),
    ('O(nlogn) 期望时间', 'O(n log n) 期望时间'),
    ('最坏情况梳排序', '最坏情况。\n梳排序'),
    ('O(n log n) 堆排序', 'O(n log n)。\n堆排序'),
    ('O(n log n) 平滑排序', 'O(n log n)。\n平滑排序'),
    ('O(n log n) 快速排序', 'O(n log n)。\n快速排序'),
    ('最坏情况\n内省排序', '最坏情况。\n内省排序'),
    ('O(n log n) 耐心排序', 'O(n log n)。\n耐心排序'),
    ('O(n+k\n)', 'O(n+k)'),
    ('O(nn!', 'O(n×n!)'),
    ('Bogo排序—O(n×n!)', 'Bogo排序—O(n×n!)，最坏情况下期望时间为无穷。'),
], '233-249')
replace('000805', [
    ('O(n3) )', 'O(n³)'), ('O(n2) )', 'O(n²)'),
    ('或O(n) )', '或 O(√n)'),
    ('硬体煎饼排序', '硬体。煎饼排序'),
    ('约n2.7 的时间均按从小到大排列\nk代表数值中的"数位"个数n代表数据规模\nm代表数据的最大值减最小值', '约 n^2.7 的时间。'),
], '250-253')
set_text('000817', '对于任何二分查找过程，搜索树的层数总为 ⌊log₂(n)+1⌋。若目标元素不在数组中，且 n 是 2 的某次幂减 1，查找必然遍历到最深层；否则，搜索可能在倒数第二层中止，只比较 ⌊log₂(n)⌋ 次。若目标元素在数组中且每个元素被搜索的概率相等，平均比较次数为 ⌊log₂(n)⌋+1−[2^{⌊log₂(n)⌋+1}−⌊log₂(n)⌋−2]/n，约为 log₂(n)−1。', '352-355')
set_text('000822', '内部路径长度的求和公式可以化简为 I(n)=Σ_{k=1}^{n}⌊log₂(k)⌋=(n+1)⌊log₂(n+1)⌋−2^{⌊log₂(n+1)⌋+1}+2。代入成功搜索的平均迭代次数，得到 T(n)=1+I(n)/n=⌊log₂(n)⌋+1−[2^{⌊log₂(n)⌋+1}−⌊log₂(n)⌋−2]/n。这与成功搜索的平均情况公式相同。', '371-387')
set_text('000825', '对于任意二叉树，外部路径长度与内部路径长度满足 E(n)=I(n)+2n。代入 I(n) 的表达式，得到 E(n)=(n+1)(⌊log₂(n)⌋+2)−2^{⌊log₂(n)⌋+1}。因此失败搜索的平均迭代次数为 T′(n)=E(n)/(n+1)=⌊log₂(n)⌋+2−2^{⌊log₂(n)⌋+1}/(n+1)。在一种二分查找实现中，每次迭代需要做一次或两次比较，以检查中间元素与目标元素的关系。', '388-401')
set_text('000838', '一种精确的量子二分查找算法在最坏情况下需要 4log₆₀₅(n)≈0.433log₂(n) 次查询。相比之下，用于搜索无序列表的格罗弗算法需要 O(√n) 次查询。将列表元素排序以提高查找效率的思想古已有之。约公元前 200 年巴比伦的 Inakibit-Anu 泥板包含约 500 个按字典序排列的六十进制数字及其倒数。1286 年完成的拉丁语词典《Catholicon》给出了完整的字母排序规则。1946 年，约翰·莫奇利首次在摩尔学院讲座中提及二分查找。', '497-506')
replace('000849', [
    ('f:X\nX', 'f:X→X'),
    ('函数F:N，这里的\n指称', '函数F:ℕ→X，这里的 ℕ 指称'),
    ('F(0)a F(n+1)f(F(n)\n)', 'F(0)=a，F(n+1)=f(F(n))。'),
    ('通过递归在\n上定义', '通过递归在 ℕ 上定义'),
    ('《数是什么且有何用？', '《数是什么且有何用？》中进行了梗概论述。'),
], '288-292')
set_text('001022', '设一训练集为 D={(p₁,t₁),…,(pₘ,tₘ)}，其中 pᵢ 表示输入，tᵢ 是对应的目标输出。由于符号函数的不连续性，如果采用标准的均方误差，所得误差函数必然是不连续的，因而基于梯度的学习算法也就不能被使用。为此，Rosenblatt 提出了感知机准则函数：E(w)=−Σ_{pᵢ∈M}(wᵀpᵢ)tᵢ，其中 M 是被当前 w 错误分类的输入向量集合。当 wᵀpᵢ≥0 时，tᵢ 为 −1；当 wᵀpᵢ<0 时，tᵢ 为 +1。故误差函数 E(w) 是一组正数的和；当训练集里所有输入都被正确分类时，它等于零。', '383-385')
set_text('001031', '梯度下降方法基于以下观察：如果实值函数 F(x) 在点 a 处可微且有定义，那么函数 F(x) 在 a 点沿着梯度相反的方向 −∇F(a) 下降最多。因而，对于足够小的 γ>0，如果 b=a−γ∇F(a)，那么 F(a)≥F(b)。从函数 F 的局部极小值的初始估计 x₀ 出发，考虑序列 x₀、x₁、x₂、…，使得 xₙ₊₁=xₙ−γₙ∇F(xₙ)，n≥0。', '156-163')
set_text('001033', '梯度下降法处理一些复杂的非线性函数会出现问题，例如 Rosenbrock 函数 f(x,y)=(1−x)²+100(y−x²)²。其最小值在 (x,y)=(1,1) 处，数值为 f(1,1)=0。但是此函数具有狭窄弯曲的山谷，最小值 (1,1) 就在这些山谷之中，并且谷底很平。优化过程之字形地向极小值点靠近，速度非常缓慢。', '174-177')
set_text('001061', '在数学与计算机科学中，尤其是在机器学习和逆问题领域中，正则化（英语：regularization）是指为解决适定性问题或过拟合而加入额外信息的过程。在机器学习和逆问题的优化过程中，正则项往往被加在目标函数当中。概括来讲，机器学习的训练过程，就是要找到一个足够好的函数 F* 用以在新的数据上进行推理。为了定义什么是“好”，人们引入了损失函数的概念。一般地，对于样本 (x⃗,y) 和模型 F，有预测值 ŷ=F(x⃗)。损失函数是定义在 ℝ×ℝ→ℝ 上的二元函数 ℓ(y,ŷ)，用来描述基准真相和模型预测值之间的差距。一般来说，损失函数是一个有下确界的函数；当基准真相和模型预测值足够接近，损失函数的值也会接近该下确界。', '316-322')

assert len(repairs) == 44
for rid, (new, lines) in repairs.items():
    item = source[rid]
    assert 'possible_formula_extraction_damage' in item['flags']
    assert new != item['record']['text'] and new.strip()
    assert abs(new.count('(') - new.count(')')) < 3, rid

NEXT.mkdir(parents=True, exist_ok=True)
for name in ('natural_zh.jsonl', 'natural_en.jsonl', 'article_en.jsonl'):
    shutil.copyfile(PREV / name, NEXT / name)
prior = [json.loads(line) for line in (PREV / 'article_zh.jsonl').open()]
ids = {x['id'] for x in prior}
assert not ids.intersection(repairs)
group_by_title = {x['source_title']: x['group_id'] for x in prior}
assert all(x['group_id'] == group_by_title[x['source_title']] for x in prior)
assert all(source[rid]['record']['source_title'] in group_by_title for rid in repairs)
for rid, (new, lines) in repairs.items():
    record = source[rid]['record'].copy()
    record['text'] = new
    record['group_id'] = group_by_title[record['source_title']]
    prior.append(record)
prior.sort(key=lambda x: int(x['id'].rsplit('-', 1)[1]))
(NEXT / 'article_zh.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in prior))

remaining = [x for x in rows if x['record']['id'] not in repairs]
(REVIEW / 'quarantined_records_round9.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in remaining))
log = []
for rid, (new, lines) in sorted(repairs.items()):
    item = source[rid]
    old_text = item['record']['text']
    title = item['record']['source_title']
    canonical_title = {'二分搜寻':'二分搜尋','简谐运动':'簡諧運動'}.get(title,title.replace(' ', '_'))
    url = 'https://zh.wikipedia.org/wiki/' + canonical_title
    log.append({'record_id':rid,'decision':'repair_formula_and_restore','source_url':url,'source_lines':lines,'source_title':title,'source_checked_on':'2026-11-07','old_text_sha256':hashlib.sha256(old_text.encode()).hexdigest(),'new_text_sha256':hashlib.sha256(new.encode()).hexdigest(),'old_text':old_text,'new_text':new,'prior_flags':item['flags']})
(REPORT / 'formula_repair_decisions_round9.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in log))
tok = Tokenizer.from_file('/Users/zhangyunjian/felixspace/rat/gpt2/tokenizer/tokenizer.json')
tok.no_truncation()
stats = {}
for name in ('natural_zh.jsonl','natural_en.jsonl','article_zh.jsonl','article_en.jsonl'):
    path = NEXT / name
    recs = [json.loads(line) for line in path.open()]
    stats[name] = {'records':len(recs),'tokens':sum(len(tok.encode(x['text'],add_special_tokens=False).ids) for x in recs),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert sum(x['records'] for x in stats.values()) == 10811
assert len(remaining) == 51
assert len({x['id'] for name in stats for x in map(json.loads,(NEXT/name).open())}) == 10811
(REPORT / 'staging_round9_manifest.json').write_text(json.dumps({'input_staging':str(PREV),'output_staging':str(NEXT),'formula_reviewed':44,'repaired_and_restored':44,'formula_still_quarantined':0,'all_risk_still_quarantined':51,'files':stats,'status':'provisional; remaining risk records and quality sample need review'},ensure_ascii=False,indent=2)+'\n')
flag_counts = collections.Counter(flag for x in remaining for flag in x['flags'])
title_urls = {x['source_title']:x['source_url'] for x in log}
title_counts = collections.Counter(x['source_title'] for x in log)
source_index = '\n'.join(f'- [{title}]({title_urls[title]})：{title_counts[title]} 条' for title in sorted(title_counts))
(REPORT / 'quality_report_round9.md').write_text(f'# littleGPT 公式核对与暂存结果（第九批）\n\n2026-11-07 对照维基百科条目中的公式与上下文，修复并恢复 44 条中文文章记录。逐条的来源 URL、网页行号、修复前后正文和 SHA-256 见 `formula_repair_decisions_round9.jsonl`。网页行号对应本次检索结果，页面更新后可能变化。\n\n- 第九批暂存训练集：10,811 条，{sum(x["tokens"] for x in stats.values()):,} 个正文 Token（不含 EOS）。\n- 44 条公式提取疑点均已修复；其它 51 条风险记录继续隔离。其标记数为：表格或标记片段 {flag_counts["possible_table_or_markup_fragment"]}、括号不平衡 {flag_counts["unbalanced_ascii_parentheses"]}、网页导航混入 {flag_counts["possible_web_navigation_mixed_with_body"]}（标记可重叠）。\n- 第八批原有 10,767 条记录保持不变；第九批新增 44 条修复记录。四份 JSONL 可解析、正文非空、无跨文件精确重复。\n- 原始语料未改动；第九批仍是暂存版，外部文章许可审查与随机质量抽检仍未完成。\n\n## 对照原文\n\n{source_index}\n')
print(json.dumps({'repaired':44,'remaining_formula':0,'remaining_risk':51,'records':10811,'tokens':sum(x['tokens'] for x in stats.values())},ensure_ascii=False))
