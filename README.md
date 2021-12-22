<!-- omit in toc -->
# 🏋️‍♂️ Revision for Codility Test on Python

## 1. References

[W3 Python Reference](https://www.w3schools.com/python/python_reference.asp)

## 2. Tasks

### 2.1. BinaryGap

Find biggest binary gap in integer N. Return 0 by default:

```python
def solution(N):
    gap = 0
    for g in bin(N).split('1')[1:-1]:
        if len(g) > gap:
            gap = len(g)
    return gap
```

### 2.2. CyclicRotation

Rotate the list. If the list is empty return the list:

```python
def solution(A, K):
    if bool(A):
        return A[-K%len(A):] + A[:-K%len(A)]
    else:
        return A
```
### 2.3. OddOccurrencesInArray

Remove odd one out in a list with an odd number of elements.

```python
def solution(A):
    seen = set()
    for i in A:
        if i not in seen:
            seen.add(i)
        else:
            seen.remove(i)
    return list(seen)[0]
```

### 2.4. FrogJmp

Help the frog reach its goal with the least number of steps necessary.

```python
def solution(X,Y,D):
    Z = Y-X
    return Z//D + (Z%D > 0)
```

### 2.5. PermMissingElem
Find the missing integer in a list.

```python
def solution(A):
    A = set(A)
    missing = 1
    while missing in A:
        missing += 1
    return missing
```

### 2.6. TapeEquilibrium

Minimize the distance between partitions of the tape.

First solution (poor performance with big N):

```python
def solution(A):
    min_d = abs(sum(A[:1]) - sum(A[1:]))
    for p in range(2,len(A)):
        d = abs(sum(A[:p]) - sum(A[p:]))
        if d < min_d:
            min_d = d

    return min_d
```

Second solution (better performance with a 100% score):

```python
def solution(A):
    l = A[0]
    r = sum(A[1:])
    min_d = abs(l-r)

    for p in range(1,len(A)):
        d = abs(l-r)
        if d < min_d:
            min_d = d
        l += A[p]
        r -= A[p]

    return min_d
```

### 2.7. FrogRiverOne

Find the min time for the frog to cross the river:

```python
def solution(X, A):
    r = set(range(1,X+1))

    for i, v in enumerate(A):
        if v in r:
            r.remove(v)
        if not bool(r):
            return i

    return -1
```

### 2.8. PermCheck

Find out if the array is a permutation of length N:

```python
def solution(A):
    p = set(range(1,len(A)+1))
    A = set(A)
    return int(A == p)
```

### 2.9. MaxCounters

Implement a list of counters with two functions: increment and max_counters.

First solution:

```python
def solution(N,A):
    c = [0]*N
    for v in A:
        if 1 <= v <= N:
            c[v-1] += 1
        else:
            c = [max(c)]*N
    return c
```

Solution with better performance:

```python
def solution(N,A):
    counters = [0]*N
    max_counter = 0
    max_to_set = 0

    for v in A:
        x = v-1
        ## lazy max function
        if x == N:
            max_to_set = max_counter
        ## increment function
        if 0 <= x < N:
            ## update counter value if (a) not maxxed or (b) maxxed previously
            counters[x] = max(counters[x]+1, max_to_set+1)
            ## update lazy max function
            max_counter = max(counters[x],max_counter)

    result = list(map(lambda c : max(c,max_to_set), counters))
    return result
```

### 2.10. MissingInteger

Find the smallest integer N that's missing in a list. Return 1 by default.

My solution:

```python
def solution(A):
    A = set(A)
    if max(A) < 0:
        return 1
    B = set(range(1,max(A)+1))
    C = B-A
    if bool(C):
        return min(C)
    else:
        return max(A)+1
```

Solution with better performance:

```python
def solution(A):
    A = set(A)
    ans = 1
    while ans in A:
       ans += 1
    return ans
```

### 2.11. PassingCars

First solution (without prefix sums):

```python
def solution(A):
    counter = 0
    for i, v in enumerate(A):
        if i == len(A):
            break
        if v == 0:
            counter = counter + sum(A[i:])
            if counter > 10**9:
                counter = -1
                break
    return counter
```

Second solution implementing prefix sums:

```python
def solution(A):
    n = len(A)
    p, max_p = prefix_sum(A)

    if max_p > 10**9:
        return -1

    counter = 0
    for i in range(0,n):
        if A[i]==0:
            counter += count_total(p,i,n-1)
            if counter > 10**9:
                return -1

    return counter

def prefix_sum(A):
    n = len(A)
    p = [0]*(n+1)
    for i in range(1,n+1):
        p[i]=p[i-1]+A[i-1]
    return (p, max(p))

def count_total(p,i,x):
    return int(p[x+1] - p[i])
```
