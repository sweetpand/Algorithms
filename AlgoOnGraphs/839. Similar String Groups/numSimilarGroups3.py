class Solution:
    def numSimilarGroups(self, A: List[str]) -> int:
        n,m=len(A),len(A[0])
        a_counter={}
        counter_a=collections.defaultdict(list)
        for a in A:
            counter=[0]*26
            for c in a:
                counter[ord(c)-ord('a')]+=1
            tmp=tuple(counter)
            a_counter[a]=tmp
            counter_a[tuple(tmp)].append(a)
        def similar(a,b):
            n=len(a)
            cnt=0
            for i in range(n):
                if a[i]!=b[i]:
                    cnt+=1
            return cnt==2
        uf={}
        def find(x):
            uf.setdefault(x,x)
            if uf[x]!=x:
                uf[x]=find(uf[x])
            return uf[x]
        def union(x,y):
            uf[find(x)]=find(y)
        for a in A:
            find(a)
            tmp=counter_a[a_counter[a]]
            if len(a)>=len(tmp):
                for b in tmp:
                    if similar(a,b):
                        union(a,b)
            else:
                for i in range(len(a)-1):
                    for j in range(i+1,len(a)):
                        b=a[:i]+a[j]+a[i+1:j]+a[i]+a[j+1:]
                        if b in a_counter:
                            union(a,b)
        return len({find(key) for key in uf})
