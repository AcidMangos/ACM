

### 素数

#### Miller-Rabin判大素数

$O(T\times\log{N})$

```cpp
#include<cstdio>
#define LL long long 
inline int read() {
    char c = getchar(); int x = 0, f = 1;
    while(c < '0' || c > '9') {if(c == '-') f = -1; c = getchar();}
    while(c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
    return x * f;
}
int N, M, Test[10] = {2, 3, 5, 7, 11, 13, 17};
int pow(int a, int p, int mod) {
    int base = 1;
    for(; p; p >>= 1, a = (1ll * a * a) % mod) 
        if(p & 1) base = (1ll * base * a) % mod;
    return base % mod;
}
bool Query(int P) {
    if(P == 1) return 0;
    int t = P - 1, k = 0;
    while(!(t & 1)) k++, t >>= 1;
    for(int i = 0; i < 4; i++) {
        if(P == Test[i]) return 1;
        LL a = pow(Test[i], t, P), nxt = a;
        for(int j = 1; j <= k; j++) {
            nxt = (a * a) % P;
            if(nxt == 1 && a != 1 && a != P - 1) return 0;
            a = nxt;
        }
        if(a != 1) return 0;
    }
    return 1;
}
main() { 
    N = read(); M = read();    
    while(M--) puts(Query(read()) ? "Yes" : "No");
}
```



#### 哥德巴赫猜想

1.2以上的偶数都能够用2个素数的和表示。

2.任何一个大于7的奇数都能被表示成三个奇质数的和。



### gcd和lcm

#### gcd

```cpp
ll gcd(ll a,ll b){
   if(b==0)return a;
    return gcd(b,a%b);
}
```



#### lcm

```cpp
ll lcm(ll a,ll b){return a / gcd(a,b) * b;}
```



#### exgcd

求出方程$ax+by=gcd(a,y)$

已知$a,b,k$,求解$ax+by=k$

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#define ll long long 
using namespace std;
ll exgcd(ll a,ll b,ll &x,ll &y)
{
	if(!b)
	{
		x=1;y=0;
		return a;
	}
	else
	{
		ll tx,ty;
		ll d=exgcd(b,a%b,tx,ty);
		x=ty;y=tx-(a/b)*ty;
		return d;
	}
}
int main()
{
	ll a,b,k,x,y;scanf("%lld%lld%lld",&a,&b,&k);
	ll d=exgcd(a,b,x,y);
	if(k%d){puts("no solution!");return 0;}
	else
	{
		x=x*k/d;
		y=(k-a*x)/b;qqqqqqqszz   s s  s ss  s s ss  s s ss  s s ss s ss s s s s
	}
	printf("%lld %lld\n",x,y);
	return 0;
}
```



求逆元

当模数为99824435时，由于mod-2不是质数，不能直接求逆元，因此使用exgcd

```cpp
ll exgcd(ll a, ll b, ll& x, ll& y){
	if (b == 0)
	{
		x = 1;
		y = 0;
		return a;
	}
	ll d = exgcd(b, a % b, x, y);
	ll t = x;
	x = y;
	y = t - (a / b) * y;
	return d;
}
//num为需要取模的数，mod为模数
ll inverse(ll num, ll mod)
{
	ll x, y;
	exgcd(num, mod, x, y);
	while (x < 0) x += mod, y -= num;
	return x;
}
```



### 数论分块

![](https://s3.bmp.ovh/imgs/2021/10/b0eb286240a9ff8b.png)



![](https://s3.bmp.ovh/imgs/2021/10/cb03916106d273b1.png)



### 欧拉函数

$φ(n)= n * \prod(1-1/p_i) $

1.若$m,n$互质，则$φ(nm)=φ(n)φ(m)  $

2.若$n$为质数，则$φ(n)=n-1$

3.若$n$为奇质数，$φ(2n)=φ(n)$

暴力解法

```cpp
#include<bits/stdc++.h>
using namespace std;
int t,n;
int main(){
    cin>>t;
    while(t--){
        cin>>n;
        int res=n;
        for(int i=2;i<=n/i;i++){
            if(n%i==0){
                while(n%i==0) n/=i;
                res=res/i*(i-1);
            }
        }
        if(n>1) res=res/n*(n-1);
        cout<<res<<endl;
    }
}

```

欧拉筛$O(n)$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e6+5;
int n,primes[N],euler[N],cnt;
bool st[N];
void get_eulers(int n){
    euler[1]=1;
    for(int i=2;i<=n;i++){
        if(!st[i]){
            primes[cnt++]=i;
            euler[i]=i-1;
        }
        for(int j=0;primes[j]<=n/i;j++){
            int t=primes[j]*i;
            st[t]=1;
            if(i%primes[j]==0){
                euler[t]=primes[j]*euler[i];
                break;
            }
            euler[t]=(primes[j]-1)*euler[i];
        }
    }
}
int main(){
    cin>>n;
    get_eulers(n);
    long long int res=0;
    for(int i=1;i<=n;i++) res+=euler[i];
    cout<<res<<endl;
}


```



### 筛法



#### 埃氏筛$O(N\log\log{N})$

```cpp
void shai(){
    for(int i = 2; i <= maxn - 5; i++){
        if(!vis[i]){
            prim[t++] = i;
            for(int j = 2; j * i <= maxn; j++)vis[i*j] = 1;
        }
    }
}
```



#### 欧拉筛$O(N)$

```cpp
void shai(){
    for(int i = 2;i <= maxn - 5; i++){
        if(!vis[i])prim[t++] = i;
        for(int j = 0; j < t && i * prim[j] <= maxn - 5; j++){
            vis[i*prim[j]] = 1;
            if(i % prim[j] == 0) break;
        }
    }
}
```



### 质因数分解$O(\sqrt{n})$

```cpp
void fen(int n){
    for(int i = 2; i * i <= n; i++){
        while(n % i == 0) fac[t++] = i,n /= i;
    }
    if(n != 1)fac[t++] = n;
}


//O(nlogn)记录因子个数
for (int i = 1; i <= 1000005; i++) {
		for (int j = i; j <= 1000005; j += i) {
			vis[j]++;
		}
	}

```



### 费马小定理

若$p$为素数，$gcd(a,p)=1$,则$a^{p-1}=1~~(mod ~~ p)$。

对于任意整数$a$，有$a^p=a~~(mod~~p)$。



### 欧拉定理

若$gcd(a,m)=1$，则$a^{φ(n)}=1~~(mod~~p)$



### 线性求逆元

```cpp
#include <iostream>
#include <algorithm>
#include <cstdio>
using namespace std;
const int maxn = 3e6+10;
int inv[maxn];

int main(){
	int n,p;
	cin >> n >> p;
	inv[1] = 1;
	cout << inv[1] << endl;
	for(int i = 2;i <= n;i++){
		inv[i] = (long long)(-p/i+p)*inv[p%i]%p;
		printf("%d\n",inv[i]);
	}
	return 0;
}
```



### FFT



```cpp
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e7 + 10;
const double pi = acos(-1.0);
struct cp {
	double x, y;
	cp(double xx = 0, double yy = 0) { x = xx, y = yy; }
}a[maxn], b[maxn];

cp operator +(cp a, cp b) { return cp(a.x + b.x, a.y + b.y); }
cp operator -(cp a, cp b) { return cp(a.x - b.x, a.y - b.y); }
cp operator *(cp a, cp b) { return cp(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

int n, m;
int len, r[maxn];
int limit = 1;

void fft(cp* A, int flag) {
	for (int i = 0; i < limit; i++)
		if (i < r[i])swap(A[i], A[r[i]]);
	for (int mid = 1; mid < limit; mid <<= 1) {
		cp wn(cos(pi / mid), flag * sin(pi / mid));
		for (int R = mid << 1, j = 0; j < limit; j += R) {
			cp w(1, 0);
			for (int k = 0; k < mid; k++, w = w * wn) {
				cp x = A[j + k], y = w * A[j + k + mid];
				A[j + k] = x + y;
				A[j + k + mid] = x - y;
			}
		}
	}
}

void mol(cp* A, cp* B, int L1, int L2) {
	limit = 1, len = 0;
	while (limit < L1 + L2 - 2)limit <<= 1, len++;
	for (int i = 0; i <= limit; i++) r[i] = (r[i >> 1] >> 1) | ((i & 1) << (len - 1));
	fft(a, -1), fft(b, -1);
	for (int i = 0; i <= limit; i++)a[i] = a[i] * b[i];
	fft(a, 1);
	for (int i = 0; i <= L1 + L2; i++) {
		a[i] = (int)(a[i].x / limit + 0.5);
	}
}

```



### NTT

```cpp
#include<iostream>
#include<algorithm>
#define ll long long
using namespace std;
const int maxn = 1e7 + 10, G = 3, Gi = 332748118;//(mod+1)/3
const ll mod = 998244353;
int n, lim = 1, len = 0;
ll r[maxn];


ll ksm(ll a, ll b) {
	ll res = 1;
	while (b > 0) {
		if (b & 1)res = res * a % mod;
		a = a * a % mod;
		b >>= 1;
	}
	return res;
}

void ntt(ll* A, int len, int flag) {
	for (int i = 0; i < len; i++)
		if (i < r[i])swap(A[i], A[r[i]]);
	for (int mid = 1; mid < len; mid <<= 1) {
		ll wn = ksm(flag == 1 ? G : Gi, (mod - 1) / (ll)(mid << 1));
		for (int j = 0; j < len; j += (mid << 1)) {
			ll w = 1;
			for (int k = 0; k < mid; k++, w = (w * wn) % mod) {
				ll x = A[j + k], y = w * A[j + k + mid] % mod;
				A[j + k] = (x + y) % mod;
				A[j + k + mid] = (x - y + mod) % mod;
			}
		}
	}
	if (flag == -1) {
		ll inv = ksm(len, mod - 2);
		for (int i = 0; i < len; i++)A[i] = (A[i] * inv + mod) % mod;
	}
}

//数组a为结果
void mol(ll* a, ll* b,int L1,int L2) {
	lim = 1, len = 0;
	while (lim < L1 + L2 - 2)lim <<= 1, len++;
	for(int i=0;i<lim;i++)r[i]= (r[i >> 1] >> 1) | ((i & 1) << (len - 1));
	ntt(a, lim, 1), ntt(b, lim, 1);
	for (int i = 0; i < lim; i++)a[i] = a[i] * b[i] % mod;
	ntt(a, lim, -1);
}
```



```cpp
//（g 是mod(r*2^k+1)的原根）

素数  r  k  g
3   1   1   2
5   1   2   2
17  1   4   3
97  3   5   5
193 3   6   5
257 1   8   3
7681    15  9   17
12289   3   12  11
40961   5   13  3
65537   1   16  3
786433  3   18  10
5767169 11  19  3
7340033 7   20  3
23068673    11  21  3
104857601   25  22  3
167772161   5   25  3
469762049   7   26  3
1004535809  479 21  3
2013265921  15  27  31
2281701377  17  27  3
3221225473  3   30  5
75161927681 35  31  3
77309411329 9   33  7
206158430209    3   36  22
2061584302081   15  37  7
2748779069441   5   39  3
6597069766657   3   41  5
39582418599937  9   42  5
79164837199873  9   43  5
263882790666241 15  44  7
1231453023109121    35  45  3
1337006139375617    19  46  3
3799912185593857    27  47  5
4222124650659841    15  48  19
7881299347898369    7   50  6
31525197391593473   7   52  3
180143985094819841  5   55  6
1945555039024054273 27  56  5
4179340454199820289 29  57  3
```





### 龟速乘（解决相乘爆LL）

```cpp
LL qadd(LL a ,LL b , LL p)
{
	LL res = 0 ;
	while(b)
	{
		if(b & 1) res = (res + a) % p;//对b进行右移运算
		a = (a + a ) % p;//改变a
		b = b>>1;
	}
	return res;
}
```



### BSGS

已知$a,b$和质数$p$，求解$a^x=b(mod~~p)$，若$x$不存在，返回-1。（保证$a,p$互质）

```cpp
ll bsgs(ll a,ll b,ll m){
	static unordered_map<ll,ll> mp;
	mp.clear();
	ll cur=1,t=sqrt(m)+1;
	for(int B=1;B<=t;B++){
		cur=cur*a%m;
		mp[b*cur%m]=B;
	}
	ll now=cur;
	for(int A=1;A<=t;A++){
		auto it = mp.find(now);
		if(it!=mp.end()) return 1ll*A*t-it->second;
	    now=now*cur%m;
	}
	return -1;
}
```



### exBSGS

$a，p$不需要互质

```cpp
ll gcd(ll a,ll b){
	return (b==0)?a:gcd(b,a%b);
}
 
ll bsgs(ll a,ll b,ll m,ll k=1){
	static unordered_map<ll,ll> mp;
	mp.clear();
	ll cur=1,t=sqrt(m)+1;
	for(int B=1;B<=t;B++){
		cur=cur*a%m;
		mp[b*cur%m]=B;
	}
	ll now=cur*k%m;
	for(int A=1;A<=t;A++){
		auto it=mp.find(now);
		if(it!=mp.end()) return 1ll*A*t-it->second;
		now=now*cur%m;
	}
	return -0x3f3f3f3f;
}
 
ll ex_bsgs(ll a,ll b,ll m,ll k=1){
	ll A=a%m,B=b%m,M=m;
	if(b==1) return 0;
	ll cur=1%m;
	for(int i=0;;i++){
		if(cur==B) return i;
		cur=cur*A%M;
		ll d=gcd(a,m);
		if(b%d) return -0x3f3f3f3f;
		if(d==1) return bsgs(a,b,m,k*a%m)+i+1;
		k=k*a/d%m;
		b/=d; 
		m/=d;
	}
}
```



### 求组合数O(N^2)

$A_{m}^n~=~n!/(n-m)!$

$C_{n}^m~=~n!/(m!*(n-m)!)$

$C_{n+1}^{m}~=~C_{n}^m ~+~C_{n}^{m-1}$

$C_{n}^0+C_{n}^1+...+C_{n}^n=2^n$

```cpp
void init() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            if (!j) c[i][j] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
        }
    }
}

```



### 矩阵相关

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N = 100;
const ll mod = 1e9 + 7;

struct matrix{
    int n;
    int m;
    ll num[N][N];
    void clear(){memset(num, 0, sizeof(num));}
};

matrix operator +(matrix a,matrix b)
{
    for (int i = 1; i <= a.n;i++)
        for (int j = 1; j <= a.m;j++)
            a.num[i][j] = (a.num[i][j] + b.num[i][j]) % mod;
    return a;
}

matrix operator *(matrix a, matrix b){
    matrix ans;
    ans.clear();
    ans.n = a.n;
    ans.m = b.m;
    for (int i = 0; i < a.n; i++)
        for (int j = 0; j < b.m; j++)
            for (int k = 0; k < a.m; k++)
                ans.num[i][j] = (ans.num[i][j] + a.num[i][k] * b.num[k][j] % mod) % mod;
    return ans;
}

bool operator !=(matrix a,matrix b)
{
    if(a.n!=b.n || a.m!=b.m)
        return 1;
    for (int i = 1; i <= a.n;i++)
        for (int j = 1; j <= a.m;j++)
            if(a.num[i][j]!=b.num[i][j])
                return 1;
    return 0;
}

matrix power(matrix a, ll x){
    matrix ans;
    ans.clear();
    ans.n = ans.m = a.n;
    for (int i = 0; i < a.n; i++)
        ans.num[i][i] = 1;
    while (x){
        if (x & 1)
            ans = ans * a;
        a = a * a;
        x >>= 1;
    }
    return ans;
}
```







### 模拟退火

  给一个N边形的所有顶点坐标，求距离该N边形所有顶点距离和最小的点。
输出该距离。



```cpp
#include <bits/stdc++.h>
#define ll long long
using namespace std;
const int maxn=100+10;
double x[maxn],y[maxn];
double ans=0;
double sumx,sumy;
int n;
double mx,my;
double calc(double px,double py)
{
    double res=0;
    for(int i=1;i<=n;i++)
    {
        res+=sqrt((px-x[i])*(px-x[i])+(py-y[i])*(py-y[i]));
    }
    return res;
}
void SA()
{
    double nowx=mx,nowy=my;
    double T=3000;
    double T_min=1e-12;
    double r=0.996;//调参
    while(T>T_min)
    {
        double nextx=nowx+(rand()*2-RAND_MAX)*T;
        double nexty=nowy+(rand()*2-RAND_MAX)*T;//随机选取的点由当前点转移而来
        double nextans=calc(nextx,nexty);//计算随机点的答案
        double delta=nextans-ans;//答案差
        if(delta<0)
        {
            ans=nextans;
            nowx=nextx,nowy=nexty;
            mx=nextx,my=nexty;
        }
        else if(exp(-delta/T)*RAND_MAX>rand())
        {
            nowx=nextx,nowy=nexty;
        }
        T*=r;//T的变化
    }
}
int main()
{
    srand(1e9+7);//随机数种子
    int t;
    scanf("%d",&t);//多组读入
    while(t--)
    {
        scanf("%d",&n);
        sumx=0,sumy=0;
        for(int i=1;i<=n;i++)
        {
            scanf("%lf%lf",&x[i],&y[i]);
            sumx+=x[i];
            sumy+=y[i];
        }
        ans=1e9;
        mx=sumx/n,my=sumy/n;//以此作为初始点
        SA();SA();SA();SA();SA();//跑几遍模拟退火提高正确率
        printf("%.0lf\n",ans);
        if(t) printf("\n");
    }
    return 0;
}
```



### 博弈论

#### 巴什博弈

我们从最简单的情景开始分析

当石子有$1−m$个时，毫无疑问，先手必胜

当石子有$m+1$个时，先手无论拿几个，后手都可以拿干净，先手必败

当石子有$m+2−2m$时，先手可以拿走几个，剩下$m+1$个，先手必胜

我们不难发现，面临$m+1$个石子的人一定失败。

这样的话两个人的最优策略一定是通过拿走石子，使得对方拿石子时还有$m+1$个

我们考虑往一般情况推广

- 设当前的石子数为$n=k∗(m+1)+r$

先手会首先拿走r个，接下来假设后手拿走x个，先手会拿走$m+1−x$个，这样博弈下去后手最终一定失败

- 设当前的石子数为$n=k∗(m+1)$

假设先手拿$x$个，后手一定会拿$m+1−x$个，这样下去先手一定失败

```cpp
#include<cstdio>
int main()
{
	int n,m;
	scanf("%d%d",&n,&m);
	if(n % (m+1) !=0) printf("first win");
	else printf("second win");
    return  0;
}
```



#### 尼姆博弈


有两个顶尖聪明的人在玩游戏，游戏规则是这样的：
有$n$堆石子，两个人可以从任意一堆石子中拿任意多个石子(不能不拿)，没法拿的人失败。问谁会胜利

当$n$堆石子的数量异或和等于0时，先手必胜，否则先手必败



#### 威佐夫博弈

有两堆石子，两个顶尖聪明的人在玩游戏，每次每个人可以从任意一堆石子中取任意多的石子或者从两堆石子中取同样多的石子，不能取得人输，分析谁会获得胜利

  这里的必输局势$（0，0）、（1，2）、（3，5）、（4，7）、（6，10）、（8，13）、（9，15）、（11，18）、（12，20）。$从这些必输局势可以发现，每组的第一个是前面没有出现的最小正整数

$a_k=[k∗(1+5–√)/2], b_k=a_k+k, k=0,1,2,3...6$

先手必败则$(y-x)*((\sqrt{5}+1)/2)~=~x$

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#define int long long 
using namespace std;
main()
{
    int a,b;
    scanf("%lld%lld",&a,&b);
    if(a>b) swap(a,b);
    int temp=abs(a-b);
    int ans=temp*(1.0+sqrt(5.0))/2.0;
    if(ans==a) printf("0");
    else 	   printf("1");
    return 0;
}
```



#### 斐波那契博弈

有一堆石子，两个顶尖聪明的人玩游戏，先取者可以取走任意多个，但不能全取完，以后每人取的石子数不能超过上个人的两倍。

**先手必败，当且仅当石子数为斐波那契数**

```cpp
#include<cstdio>
#include<map>
int fib[233],x;
std::map<int,bool>mp;
int main()
{
	fib[1]=1;fib[2]=1;
	for(int i=3;i<=50;i++) fib[i]=fib[i-1]+fib[i-2],mp[fib[i]]=1;
	while(scanf("%d",&x)&&x!=0)
		puts(mp[x]==1?"Second win":"First win");
    return 0;
}

```



![](https://s3.bmp.ovh/imgs/2021/10/8fc095109a476d50.png)



#### SG函数

```cpp
int F[MAXN];//可以转移的状态集合，一般题目会给出 
int S[MAXN];//表示该点可以转移到的状态有哪些 
int SG[MAXN];//该点的SG值 
void GetSG()
{
	for(int i=1;i<=N;i++)//枚举DAG中所有点 
	{
		memset(S,0,sizeof(S));//初始化
		for(int j=1;j<=limit&&F[j]<=i;j++)//limit表示转移的集合的大小
			S[SG[i-F[j]]]=1; 
		for(int j=0;;j++)
			if(!S[j])
				{SG[i]=i;break;}//根据定义计算SG函数 
	}
}
```



### 中国剩余定理

![](https://s3.bmp.ovh/imgs/2021/10/c431de8a2bb743d2.png)



情况1，m之间互质。

```cpp
1
//扩展欧几里得模板
int ex_gcd(int a,int b,int &x,int &y){
    int d;
    if(b == 0){
        x = 1;
        y = 0;
        return a;
    }
    d = ex_gcd(b,a%b,y,x);
    y -= a / b * x;
    return d;
}
int Chinese_Remainder(int a[],int prime[],int len){
    int i,d,R,y,M,m = 1,sum = 0;
    //计算所有除数的积，也就是所有除数的最小公倍数m
    for(i = 0; i < len; i++)
        m *= prime[i];
    //计算符合所有条件的数
    for(i = 0; i < len; i++){
        M = m / prime[i];//计算除去本身的所有除数的积M
        d = ex_gcd(M,prime[i],R,y);
        sum = (sum + R * M * a[i]) % m;
    }
    return (m + sum % m) % m;//满足所有方程的最小解
}

2
#include <cstdio>
#define ll long long int
using namespace std;
int N;
ll a[15], b[15], Mi[15], X, M = 1;
ll exgcd(ll a, ll b, ll &x, ll &y)
{
	if(b==0)
	{
		x = 1; y = 0;
		return a;
	}
	ll r = exgcd(b, a%b, x, y);
	ll temp = y;
	y = x-(a/b)*y;
	x = temp;
	return r;
}
int main()
{
	scanf("%d", &N);
	for(int i=1;i<=N;i++)
	{
		scanf("%d%d", &a[i], &b[i]);
		M *= a[i];
	}
	for(int i=1;i<=N;i++)
	{
		Mi[i] = M/a[i];
		ll x = 0, y = 0;
		exgcd(Mi[i], a[i], x, y);         //exgcd求逆元
		X += b[i]*Mi[i]*(x<0 ? x+a[i] : x);
	}
	printf("%lld\n", X%M);
	return 0;
}
```



情况2，模数不一定互质

```cpp
 //有时候可能会用到lcm来求解多组解，因为你求出来的是最小解，所以你就可以通过不断加lcm求解多组解
    int lcm;
    int china2(int num){//不互质的中国剩余定理
        int m1=m[0],a1=a[0],m2,a2,k1,k2,x0,gcd,c;//这里的x0相当于推导过程中的x1
        lcm=m[0];
        for(int i=1;i<num;i++){
            m2=m[i],a2=a[i];
            c=a2-a1;
            gcd=exgcd(m1,m2,k1,k2);//解得：n1*k1+n2*k2=gcd(n1,n2)
            lcm=lcm*m[i]/Gcd(lcm,m[i]);//通过这个循环求解出所有mod的最大公约数
            if(c%gcd){
               flag=1;//!!!!!!china也可以求解出为0的值，所以为0不一定就是无解，所以你要通过flag来判断是否无解。
               return 0;//无解
            }
            x0=c/gcd*k1;//n1*x0+n2*(c/gcd*k2)=c  PS:k1/gcd*c错误！
            int t=m2/gcd;
            x0=(x0%t+t)%t;//求n1*x0+n2*y=c的x0的最小解
            a1+=m1*x0;
            m1=m2/gcd*m1;
        }
        return a1;
    }
```



### 计算几何

#### 浮点数处理

```cpp
#include <cstdio>
#include <cmath>

using namespace std;

const double pi = acos(-1.0);
const double inf = 1e100;
const double eps = 1e-6;
int sgn(double d){
    if(fabs(d) < eps)
        return 0;
    if(d > 0)
        return 1;
    return -1;
}
int dcmp(double x, double y){
    if(fabs(x - y) < eps)
        return 0;
    if(x > y)
        return 1;
    return -1;
}
int main() {
    double x = 1.49999;
    int fx = floor(x);//向下取整函数
    int cx = ceil(x);//向上取整函数
    int rx = round(x);//四舍五入函数
    printf("%f %d %d %d\n", x, fx, cx, rx);
    //输出结果 1.499990 1 2 1
    return  0 ;
}
```



#### 点与向量

```cpp
struct Point{
    double x, y;
    Point(double x = 0, double y = 0):x(x),y(y){}
};
typedef Point Vector;
Vector operator + (Vector A, Vector B){
    return Vector(A.x+B.x, A.y+B.y);
}
Vector operator - (Point A, Point B){
    return Vector(A.x-B.x, A.y-B.y);
}
Vector operator * (Vector A, double p){
    return Vector(A.x*p, A.y*p);
}
Vector operator / (Vector A, double p){
    return Vector(A.x/p, A.y/p);
}
bool operator < (const Point& a, const Point& b){
    if(a.x == b.x)
        return a.y < b.y;
    return a.x < b.x;
}
const double eps = 1e-6;
int sgn(double x){
    if(fabs(x) < eps)
        return 0;
    if(x < 0)
        return -1;
    return 1;
}
bool operator == (const Point& a, const Point& b){
    if(sgn(a.x-b.x) == 0 && sgn(a.y-b.y) == 0)
        return true;
    return false;
}
double Dot(Vector A, Vector B){//点乘
    return A.x*B.x + A.y*B.y;
}
double Length(Vector A){//长度
    return sqrt(Dot(A, A));
}
double Angle(Vector A, Vector B){//两向量夹角
    return acos(Dot(A, B)/Length(A)/Length(B));
}
double Cross(Vector A, Vector B){//叉乘
    return A.x*B.y-A.y*B.x;
}
double Area2(Point A, Point B, Point C){//计算两向量构成的平行四边形有向面积
    return Cross(B-A, C-A);
}

// 计算向量逆时针旋转后的向量
Vector Rotate(Vector A, double rad){//rad为弧度 且为逆时针旋转的角
    return Vector(A.x*cos(rad)-A.y*sin(rad), A.x*sin(rad)+A.y*cos(rad));
}

Vector Normal(Vector A){//向量A左转90°的单位法向量
    double L = Length(A);
    return Vector(-A.y/L, A.x/L);
}

//判断折线bc是不是向ab 的逆时针方向（左边）转向

bool ToLeftTest(Point a, Point b, Point c){
    return Cross(b - a, c - b) > 0;
}
```



#### 点和线

```cpp
struct Line{//直线定义
    Point v, p;
    Line(Point v, Point p):v(v), p(p) {}
    Point point(double t){//返回点P = v + (p - v)*t
        return v + (p - v)*t;
    }
};
//计算两直线交点
//调用前需保证 Cross(v, w) != 0
Point GetLineIntersection(Point P, Vector v, Point Q, Vector w){
    Vector u = P-Q;
    double t = Cross(w, u)/Cross(v, w);
    return P+v*t;
}
//点P到直线AB距离公式
double DistanceToLine(Point P, Point A, Point B){
    Vector v1 = B-A, v2 = P-A;
    return fabs(Cross(v1, v2)/Length(v1));
}//不去绝对值，得到的是有向距离
//点P到线段AB距离公式
double DistanceToSegment(Point P, Point A, Point B){
    if(A == B)
        return Length(P-A);
    Vector v1 = B-A, v2 = P-A, v3 = P-B;
    if(dcmp(Dot(v1, v2)) < 0)
        return Length(v2);
    if(dcmp(Dot(v1, v3)) > 0)
        return Length(v3);
    return DistanceToLine(P, A, B);
}
//点P在直线AB上的投影点
Point GetLineProjection(Point P, Point A, Point B){
    Vector v = B-A;
    return A+v*(Dot(v, P-A)/Dot(v, v));
}
//判断p点是否在线段a1a2上
bool OnSegment(Point p, Point a1, Point a2){
    return dcmp(Cross(a1-p, a2-p)) == 0 && dcmp(Dot(a1-p, a2-p)) < 0;
}
//判断两线段是否相交
bool SegmentProperIntersection(Point a1, Point a2, Point b1, Point b2){
    double c1 = Cross(a2-a1, b1-a1), c2 = Cross(a2-a1, b2-a1);
    double c3 = Cross(b2-b1, a1-b1), c4 = Cross(b2-b1, a2-b1);
    //if判断控制是否允许线段在端点处相交，根据需要添加
    if(!sgn(c1) || !sgn(c2) || !sgn(c3) || !sgn(c4)){
        bool f1 = OnSegment(b1, a1, a2);
        bool f2 = OnSegment(b2, a1, a2);
        bool f3 = OnSegment(a1, b1, b2);
        bool f4 = OnSegment(a2, b1, b2);
        bool f = (f1|f2|f3|f4);
        return f;
    }
    return (sgn(c1)*sgn(c2) < 0 && sgn(c3)*sgn(c4) < 0);
}

//不允许在顶点处相交
bool SegmentProperIntersection(Point a1, Point a2, Point b1, Point b2){
    double c1 = Cross(a2 - a1, b1 - a1), c2 = Cross(a2 - a1, b2 - a1);
    double c3 = Cross(b2 - b1, a1 - b1), c4 = Cross(b2 - b1, a2 - b1);
    return (sgn(c1)*sgn(c2) < 0 && sgn(c3)*sgn(c4) < 0);
}


```



#### 多边形

```cpp
//多边形有向面积
double PolygonArea(Point* p, int n){//p为端点集合，n为端点个数
    double s = 0;
    for(int i = 1; i < n-1; ++i)
        s += Cross(p[i]-p[0], p[i+1]-p[0]);
    return s;
}
//判断点是否在多边形内，若点在多边形内返回1，在多边形外部返回0，在多边形上返回-1
int isPointInPolygon(Point p, vector<Point> poly){
    int wn = 0;
    int n = poly.size();
    for(int i = 0; i < n; ++i){
        if(OnSegment(p, poly[i], poly[(i+1)%n])) return -1;
        int k = sgn(Cross(poly[(i+1)%n] - poly[i], p - poly[i]));
        int d1 = sgn(poly[i].y - p.y);
        int d2 = sgn(poly[(i+1)%n].y - p.y);
        if(k > 0 && d1 <= 0 && d2 > 0) wn++;
        if(k < 0 && d2 <= 0 && d1 > 0) wn--;
    }
    if(wn != 0)
        return 1;
    return 0;
}
```



#### 圆

```cpp
struct Circle{
    Point c;
    double r;
    Circle(Point c, double r):c(c), r(r) {}
    Point point(double a){//通过圆心角求坐标
        return Point(c.x + cos(a)*r, c.y + sin(a)*r);
    }
};

//求圆与直线交点
int getLineCircleIntersection(Line L, Circle C, double& t1, double& t2, vector<Point>& sol){
    double a = L.v.x, b = L.p.x - C.c.x, c = L.v.y, d = L.p.y - C.c.y;
    double e = a*a + c*c, f = 2*(a*b + c*d), g = b*b + d*d - C.r*C.r;
    double delta = f*f - 4*e*g;//判别式
    if(sgn(delta) < 0)//相离
        return 0;
    if(sgn(delta) == 0){//相切
        t1 = -f /(2*e);
        t2 = -f /(2*e);
        sol.push_back(L.point(t1));//sol存放交点本身
        return 1;
    }
    //相交
    t1 = (-f - sqrt(delta))/(2*e);
    sol.push_back(L.point(t1));
    t2 = (-f + sqrt(delta))/(2*e);
    sol.push_back(L.point(t2));
    return 2;
}

//两圆相交面积
double AreaOfOverlap(Point c1, double r1, Point c2, double r2){
    double d = Length(c1 - c2);
    if(r1 + r2 < d + eps)
        return 0.0;
    if(d < fabs(r1 - r2) + eps){
        double r = min(r1, r2);
        return pi*r*r;
    }
    double x = (d*d + r1*r1 - r2*r2)/(2.0*d);
    double p = (r1 + r2 + d)/2.0;
    double t1 = acos(x/r1);
    double t2 = acos((d - x)/r2);
    double s1 = r1*r1*t1;
    double s2 = r2*r2*t2;
    double s3 = 2*sqrt(p*(p - r1)*(p - r2)*(p - d));
    return s1 + s2 - s3;
}

```



#### 计算几何代码详解



#### 俊杰的板子+自己的注释和修改

```cpp
#include<bits/stdc++.h>
#define ll long long
#define _T ll
using namespace std;

constexpr double eps = 1e-8;
//constexpr long long eps=0;
constexpr double PI = 3.1415926535897932384l;

//数据范围超过1e9使用long double

template<typename T> struct point
{
	T x, y;

	bool operator==(const point& a) const { return (abs(x - a.x) <= eps && abs(y - a.y) <= eps); }
	bool operator<(const point& a) const { if (abs(x - a.x) <= eps) return y < a.y - eps; return x < a.x - eps; }
	point operator+(const point& a) const { return { x + a.x,y + a.y }; }
	point operator-(const point& a) const { return { x - a.x,y - a.y }; }
	point operator-() const { return { -x,-y }; }
	point operator*(const T k) const { return { k * x,k * y }; }
	point operator/(const T k) const { return { x / k,y / k }; }
	T operator*(const point& a) const { return x * a.x + y * a.y; } //点乘 Dot
	T operator^(const point& a) const { return x * a.y - y * a.x; } //叉乘 Cross
	// 1:左  -1:右  0:线上 （点和向量）
	//auto的原理就是根据后面的值，来自己推测前面的类型是什么。
	//return *this返回当前对象， return this返回当前对象的地址（指向当前对象的指针）
	int toleft(const point& a) const { const auto t = (*this) ^ a; return (t > eps) - (t < -eps); }
	T len2() const { return (*this) * (*this); }
	T dis2(const point& a) const { return (a - (*this)).len2(); }
	double len() const { return sqrt(len2()); }//向量长度
	double dis(const point& a) const { return sqrt(dis2(a)); }//点距
	//夹角
	double ang(const point& a) const { return acos(max(-1.0, min(1.0, ((*this) * a) / (len() * a.len())))); }
	//逆时针旋转rad
	point rot(const double rad) const { return { x * cos(rad) - y * sin(rad),x * sin(rad) + y * cos(rad) }; }
};

using Point = point<_T>;


bool argcmp(const Point& a, const Point& b) {
	auto quad = [](const Point& a) {
		if (a.y < -eps)return 1;
		if (a.y > eps)return 4;
		if (a.x < -eps)return 5;
		if (a.x > eps)return 3;
		return 2;
	};
	//1:X轴下面，2：原点，3：X轴正半轴，4：X轴上面，5：X轴负半轴
   //可自行定义
	int qa = quad(a), qb = quad(b);
	if (qa != qb)return qa < qb;
	auto t = a ^ b;
	//极轴排序
	//if (abs(t) <= eps)return a * a < b* b - eps;
	return t > eps;
}

/*
//重载写法
struct argcmp
{
	bool operator()(const Point& a, const Point& b) const
	{
		const auto quad = [](const Point& a)
		{
			if (a.y < -eps) return 1;
			if (a.y > eps) return 4;
			if (a.x < -eps) return 5;
			if (a.x > eps) return 3;
			return 2;
		};
		const int qa = quad(a), qb = quad(b);
		if (qa != qb) return qa < qb;
		const auto t = a ^ b;
		//if (abs(t)<=eps) return a*a<b*b-eps;
		return t > eps;
	}
};
sort(ps.begin(), ps.end(), [](const pair<Point, int>& a, const pair<Point, int>& b)
	{return argcmp()(a.first, b.first); });
*/

template<typename T> struct line
{
	point<T> p, v;//p为点，v为方向

	bool operator==(const line& a) const { return v.toleft(a.v) == 0 && v.toleft(p - a.p) == 0; }
	//同point中的toleft，点和直线的关系
	int toleft(const point<T>& a) const { return v.toleft(a - p); }
	//直线交点
	point<T> inter(const line& a) const { return p + v * ((a.v ^ (p - a.p)) / (v ^ a.v)); }
	//点到直线的距离
	double dis(const point<T>& a) const { return abs(v ^ (a - p)) / v.len(); }
	//a在直线上的投影点
	point<T> proj(const point<T>& a) const { return p + v * ((v * (a - p)) / (v * v)); }
};

using Line = line<_T>;


template<typename T>struct segment {
	point<T>a, b;//线段端点

	//点和线段的关系
	//-1:点在线段端点，1:点在线段上，0:点不在线段上
	int is_on(const point<T>& p) const
	{
		if (p == a || p == b) return -1;//点在线段端点上
		return (p - a).toleft(p - b) == 0 && (p - a) * (p - b) < -eps;//点在线段上
	}

	//线段与直线的关系
	//-1：线段端点在直线上，1:直线与线段相交，0:直线与线段不相交
	int is_inter(const line<T>& l) const
	{
		if (l.toleft(a) == 0 || l.toleft(b) == 0) return -1;//线段端点在直线上
		return l.toleft(a) != l.toleft(b);//
	}

	//线段与线段的关系
	//-1:线段a端点在线段b上，1:线段与线段相交，0:线段与线段不相交
	int is_inter(const segment<T>& s) const
	{
		if (is_on(s.a) || is_on(s.b) || s.is_on(a) || s.is_on(b)) return -1;
		const line<T> l{ a,b - a }, ls{ s.a,s.b - s.a };
		return l.toleft(s.a) * l.toleft(s.b) == -1 && ls.toleft(a) * ls.toleft(b) == -1;
	}

	//点到线段的距离
	double dis(const point<T>& p) const
	{
		if ((p - a) * (b - a) < -eps || (p - b) * (a - b) < -eps) return min(p.dis(a), p.dis(b));
		const line<T> l{ a,b - a };
		return l.dis(p);
	}

	//线段到线段的距离
	double dis(const segment<T>& s) const
	{
		if (is_inter(s)) return 0;
		return min({ dis(s.a),dis(s.b),s.dis(a),s.dis(b) });
	}
};

using Segment = segment<_T>;

//多边形
template<typename T> struct polygon//逆时针方向保存点
{
	vector<Point> p;
	//返回下一个点和前一个点
	//size_t可理解为unsigned int,应为数组下标类型
	inline size_t nxt(const size_t i) const { return i == p.size() - 1 ? 0 : i + 1; }
	inline size_t pre(const size_t i) const { return i == 0 ? p.size() - 1 : i - 1; }

	//回转数计算，回转数为0，点在多边形外
	pair<bool, int> winding(const Point& a) const
	{
		int cnt = 0;
		for (size_t i = 0; i < p.size(); i++)
		{
			Point u = p[i], v = p[nxt(i)];
			if (abs((a - u) ^ (a - v)) <= eps && (a - u) * (a - v) <= eps) return { true,0 };//点在边上
			if (abs(u.y - v.y) <= eps) continue;
			Line uv = { u,v - u };
			if (u.y < v.y - eps && uv.toleft(a) <= 0) continue;//该射线与边无交点
			if (u.y > v.y + eps && uv.toleft(a) >= 0) continue;
			//计算wn数
			if (u.y < a.y - eps && v.y >= a.y - eps) cnt++;
			if (u.y >= a.y - eps && v.y < a.y - eps) cnt--;
		}
		return { false,cnt };
	}

	//边长
	double circ() const
	{
		double sum = 0;
		for (size_t i = 0; i < p.size(); i++) sum += p[i].dis(p[nxt(i)]);
		return sum;
	}
	//面积
	T area2() const
	{
		T sum = 0;
		for (size_t i = 0; i < p.size(); i++) sum += p[i] ^ p[nxt(i)];
		return abs(sum);
	}
};

using Polygon = polygon<_T>;



//凸包
template<typename T> struct convex : polygon<int> //逆时针方向保存点
{

	//闵可夫斯基和
	//凸包按向量移动
	//解决凸包之间的关系问题，eg，求最大最小距离，A-B的点离原点的距离 O(n)
	//可能会出现三点共线
	convex operator+(const convex& c) const
	{
		const auto& p = this->p;
		vector<Segment> e1(p.size()), e2(c.p.size()), edge(p.size() + c.p.size());
		vector<point<T>> res; res.reserve(p.size() + c.p.size());
		const auto cmp = [](const Segment& u, const Segment& v) {return argcmp()(u.b - u.a, v.b - v.a); };
		for (size_t i = 0; i < p.size(); i++) e1[i] = { p[i],p[this->nxt(i)] };
		for (size_t i = 0; i < c.p.size(); i++) e2[i] = { c.p[i],c.p[c.nxt(i)] };
		rotate(e1.begin(), min_element(e1.begin(), e1.end(), cmp), e1.end());
		rotate(e2.begin(), min_element(e2.begin(), e2.end(), cmp), e2.end());
		merge(e1.begin(), e1.end(), e2.begin(), e2.end(), edge.begin(), cmp);
		const auto check = [](const vector<point<T>>& res, const point<T>& u)
		{
			const auto back1 = res.back(), back2 = *prev(res.end(), 2);
			return (back1 - back2).toleft(u - back1) == 0 && (back1 - back2) * (u - back1) >= -eps;
		};
		auto u = e1[0].a + e2[0].a;
		for (const auto& v : edge)
		{
			while (res.size() > 1 && check(res, u)) res.pop_back();
			res.push_back(u);
			u = u + v.b - v.a;
		}
		if (res.size() > 1 && check(res, res[0])) res.pop_back();
		return { res };
	}

	//旋转卡尺
	template<typename F> void rotcaliper(const F& func) const
	{
		const auto& p = this->p;
		const auto area = [](const point<T>& u, const point<T>& v, const point<T>& w) {return abs((w - u) ^ (w - v)); };
		for (size_t i = 0, j = 1; i < p.size(); i++)
		{
			const auto nxti = this->nxt(i);
			func(p[i], p[nxti], p[j]);
			while (area(p[this->nxt(j)], p[i], p[nxti]) >= area(p[j], p[i], p[nxti]))
			{
				j = this->nxt(j);
				func(p[i], p[nxti], p[j]);
			}
		}
	}

	//调用旋转卡尺，求凸包直径(如何使用旋转卡尺函数的样例
	T diameter2() const
	{
		const auto& p = this->p;
		if (p.size() == 1) return 0;
		if (p.size() == 2) return p[0].dis2(p[1]);
		T ans = 0;
		auto func = [&](const point<T>& u, const point<T>& v, const point<T>& w) {ans = max({ ans,w.dis2(u),w.dis2(v) }); };
		rotcaliper(func);
		return ans;
	}

	//求区间面积
	//求区间面积，以第一个点为顶点凸包分成多个三角形
	vector<T> sum;

	void get_sum()
	{
		const auto& p = this->p;
		vector<T> a(p.size());
		for (size_t i = 0; i < p.size(); i++) a[i] = p[this->pre(i)] ^ p[i];
		sum.resize(p.size());
		partial_sum(a.begin(), a.end(), sum.begin());//前缀和
	}

	T query_sum(const size_t l, const size_t r) const
	{
		const auto& p = this->p;
		if (l <= r) return sum[r] - sum[l] + (p[r] ^ p[l]);
		return sum[p.size() - 1] - sum[l] + sum[r] + (p[r] ^ p[l]);
	}
	T query_sum() const { return sum.back(); }

	//点是否在凸多边形内 log
	//-1:在线上 ,0:不在 ,1:在
	int is_in(const point<T>& a) const
	{
		const auto& p = this->p;
		if (p.size() == 1) return a == p[0] ? -1 : 0;
		if (p.size() == 2) return segment<T>{p[0], p[1]}.is_on(a) ? -1 : 0;
		if (a == p[0]) return -1;
		if ((p[1] - p[0]).toleft(a - p[0]) == -1 || (p.back() - p[0]).toleft(a - p[0]) == 1) return 0;
		const auto cmp = [&](const Point& u, const Point& v) {return (u - p[0]).toleft(v - p[0]) == 1; };
		const size_t i = lower_bound(p.begin() + 1, p.end(), a, cmp) - p.begin();
		if (i == 1) return segment<T>{p[0], p[i]}.is_on(a) ? -1 : 0;
		if (i == p.size() - 1 && segment<T>{p[0], p[i]}.is_on(a)) return -1;
		if (segment<T>{p[i - 1], p[i]}.is_on(a)) return -1;
		return (p[i] - p[i - 1]).toleft(a - p[i - 1]) > 0;
	}

	template<typename F> size_t extreme(const F& dir) const//nlogn
	{
		const auto& p = this->p;
		const auto check = [&](const size_t i) {return dir(p[i]).toleft(p[this->nxt(i)] - p[i]) >= 0; };
		const auto dir0 = dir(p[0]); const auto check0 = check(0);
		if (!check0 && check(p.size() - 1)) return 0;
		const auto cmp = [&](const Point& v)
		{
			const size_t vi = &v - p.data();
			const auto checkv = check(vi);
			const auto t = dir0.toleft(v - p[0]);
			return checkv ^ (checkv == check0 && ((!check0 && t <= 0) || (check0 && t < 0)));
		};
		return partition_point(p.begin(), p.end(), cmp) - p.begin();
	}

	//过一点求凸包切线，返回切点下标
	pair<size_t, size_t> tangent(const point<T>& a) const //!is_in(a)，保证点不在多边形内才能调用
	{
		const size_t i = extreme([&](const point<T>& u) {return u - a; });
		const size_t j = extreme([&](const point<T>& u) {return a - u; });
		return { i,j };
	}

	//求与某直线平行的切线，返回切点下标(结合to_left可用于判断直线是否与凸包相交
	pair<size_t, size_t> tangent(const line<T>& a) const
	{
		const size_t i = extreme([&](...) {return a.v; });
		const size_t j = extreme([&](...) {return -a.v; });
		return { i,j };
	}
};

using Convex = convex<_T>;


//返回一个点集的凸包
Convex convexhull(vector<Point> p)//逆时针方向保存点
{
	vector<Point> st;
	sort(p.begin(), p.end());
	const auto check = [](const vector<Point>& st, const Point& u)
	{
		const auto back1 = st.back(), back2 = *(prev(st.end(), 2));
		return (back1 - back2).toleft(u - back2) <= 0;
	};
	for (const Point& u : p)
	{
		while (st.size() > 1 && check(st, u)) st.pop_back();
		st.push_back(u);
	}
	size_t k = st.size();
	p.pop_back(); reverse(p.begin(), p.end());
	for (const Point& u : p)
	{
		while (st.size() > k && check(st, u)) st.pop_back();
		st.push_back(u);
	}
	st.pop_back();
	Convex C; C.p = st;
	return C;
}


```







### 泰勒公式



![](https://s3.bmp.ovh/imgs/2021/12/62f1429382c736ef.jpg)



### 等价无穷小

![](https://s3.bmp.ovh/imgs/2021/12/9edb01fc3f5467a0.jpg)





  



### 三角公式及定理



$a^2=b^2+c^2-2bccosA$



r 为外接圆半径

$\frac{a}{sinA}=\frac{b}{sinB}=\frac{c}{sinC}=2*r=D$









## 图论



### 前向星

```c++
int head[maxn], tot = 0,；
struct edge { int v, w, nt; }G[maxn];

Memset(head,-1,sizeof(head))

void add(int u, int v, int w) {
	G[tot].v = v;
	G[tot].w = w;
	G[tot].nt = head[u];
	head[u] = tot++;
}

```





### 最短路

![](https://s3.bmp.ovh/imgs/2021/11/0323a7440e43ca46.jpg)



#### dijkstra(堆优化)



#### SPFA



#### Floyd

$O(n^3)$暴力枚举中间节点

```cpp
void Floyd()
{
    memset(map,0x3f,sizeof(map));
    for(int i=1;i<=n;i++)
        map[i][i]=0;
    for(int k=1;k<=n;k++)//顺序不要反
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                map[i][j]=min(map[i][k]+map[k][j],map[i][j]);
}
```



### 拓扑排序

1.有向无环图才有拓扑排序

![](https://s3.bmp.ovh/imgs/2021/12/726249d438c7ffae.png)



排序后

![](https://s3.bmp.ovh/imgs/2021/12/9a8f57e9c25b188d.png)





①在有向无环图中找到没有入度的点作为初始点，放进队列。

②取出队列的第一个点，放入存储拓扑排序的队列遍历其后继点，若该后继点的入度-1=0，则放进队列中。

③循环操作2直至队列为空。

 

若存放拓扑排序的队列未放入所有的点，则说明剩下的点为环。



应用：

1.有向无环图拓扑排序，如知道点的关系，求排序。

2.判环或找环

 

```c++
void sol() {
	for (int i = 1; i <= n; i++) {
		if (in[i] == 0)que.push(i);
	}
	while (!que.empty()) {
		int now = que.front(); que.pop();
		sta.push_back(now);
		for (int i = head[now]; ~i; i = G[i].nt) {
			int v = G[i].v;
			in[v]--;
			if (!in[v])que.push(v);
		}
	}
}


```



### tarjan算法（找强连通分量，缩点）



![](https://s3.bmp.ovh/imgs/2021/12/3e5d3c0455a46923.png)

类似于dfs，不停给每个点标记（cnt++）并放入栈内，若找到找过的点，把栈内当前点到找过的

点之间染色。

![](https://s3.bmp.ovh/imgs/2021/12/23eb1ffb9c867546.png)





## 字符串

### KMP

复杂度 O(S+T)

```cpp
string s, t;
ll nxt[maxn];
vector<ll>vec;

void get_nxt(int len) {
	for (int i = 2; i <= len; i++) {
		nxt[i] = nxt[i - 1];
		while (nxt[i] && t[i] != t[nxt[i] + 1])nxt[i] = nxt[nxt[i]];
		nxt[i] += (t[i] == t[nxt[i] + 1]);
	}
}

void kmp(int L, int len) {
	for (ll i = 1, j = 1; i <= L;) {
		while (j != 1 && s[i] != t[j])j = nxt[j - 1] + 1;
		if (s[i] == t[j])j++, i++;
		else i++;
		if (j == len + 1) {
			vec.push_back(i - j + 1ll);
			j = nxt[len] + 1;
		}
	}
}

```



### Hash



$H(s[l,r])=(S[l]*Base^{r-l}+S[l+1]*Base^{r-l-1}+...+S[r])$%Mod

令 $F(i) = H(pre[i])$

$F(l-1)=(S[1]*Base^{l-2}+S[2]*Base^{l-3}+...+S[l-1])$% Mod

$~~~~~~F(r)=(S[1]*Base^{r-1}+S[2]*Base^{r-2}+...+S[r])$% Mod

$H(s[l,r])=F(r)-F(l-1)*Base^{r-l+1}$



#### 质数表

模数一般用$10^9$和$10^{10}$的质数

![caa10c00eb697d65c433babf1176920b.png](https://images.gxuca.team/images/2022/02/23/caa10c00eb697d65c433babf1176920b.png)

https://www.cnblogs.com/ljxtt/p/13514346.html





