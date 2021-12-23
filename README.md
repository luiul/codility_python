<!-- omit in toc -->
# 🏋️‍♂️ Revision for Codility Test on Python

<!-- omit in toc -->
## Description
Repository for storing codility functions from exercises for future reference.

<!-- omit in toc -->
## Table of Contents
- [1. References](#1-references)
- [2. Tasks](#2-tasks)
  - [2.1. BinaryGap](#21-binarygap)
  - [2.2. CyclicRotation](#22-cyclicrotation)
  - [2.3. OddOccurrencesInArray](#23-oddoccurrencesinarray)
  - [2.4. FrogJmp](#24-frogjmp)
  - [2.5. PermMissingElem](#25-permmissingelem)
  - [2.6. TapeEquilibrium](#26-tapeequilibrium)
  - [2.7. FrogRiverOne](#27-frogriverone)
  - [2.8. PermCheck](#28-permcheck)
  - [2.9. MaxCounters](#29-maxcounters)
  - [2.10. MissingInteger](#210-missinginteger)
  - [2.11. PassingCars](#211-passingcars)
  - [2.12. CountDiv](#212-countdiv)
  - [2.13. GenomicRangeQuery](#213-genomicrangequery)
  - [2.14. MinAvgTwoSlice](#214-minavgtwoslice)

## 1. References

[W3 Python Reference](https://www.w3schools.com/python/python_reference.asp)

## 2. Tasks

### 2.1. BinaryGap

Find biggest binary gap in integer N. Return 0 by default.

```python
def solution(N):
    gap = 0
    for g in bin(N).split('1')[1:-1]:
        if len(g) > gap:
            gap = len(g)
    return gap
```

### 2.2. CyclicRotation

Rotate the list. If the list is empty return the list.

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

Second solution (better performance with a 100% score).

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

Find the min time for the frog to cross the river.

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

Find out if the array is a permutation of length N.

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

Solution with better performance.

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

Solution with better performance.

```python
def solution(A):
    A = set(A)
    ans = 1
    while ans in A:
       ans += 1
    return ans
```

### 2.11. PassingCars

First solution (without prefix sums).

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

Second solution implementing prefix sums.

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

### 2.12. CountDiv

Find number of divisible number by K in the range [A,B]. Edge cases: A = B.

Naive solution:

```python
def solution(A, B, K):
    counter = 0
    for i in range(A,B+1):
        if i % K == 0:
            counter+=1
    return counter
```

Solution with better performance.

```python
def solution(A, B, K):
    smallest_i = 0
    if A == B:
        return int(A % K == 0)
    for i in range(A,B+1):
        if i % K == 0:
            smallest_i = i
            break
    c = len(range(smallest_i,B+1))//K + (len(range(smallest_i,B+1))%K>0)
    return c
```

### 2.13. GenomicRangeQuery

Find min weight in a slice of a string.

First solution is over-complicated because task was misunderstood.

```python
def solution(S, P, Q):
    # get list of weights
    WS = string_to_int_list(S)
    # print(WS)

    # get prefix sum
    # PRE = prefix_sum(WS)
    # print(PRE)

    # make tuples with start and end points
    T = make_tuples(P,Q)
    # print(T)

    # get slice totals and store in list
    min_weights = []
    for start, end in T:
        min_weight = min(WS[start:end+1])
        min_weights.append(min_weight)

    return min_weights

def string_to_int_list(S):
    # string -> list of char
    SI = [char for char in S]
    # dictionary of weights
    w = {'A':1, 'C':2, 'G':3, 'T':4}
    # list of char -> list of weights
    WS = list(map(w.get, SI))
    # return list of weights
    return WS

def prefix_sum(S):
    n = len(S)
    P = [0]*(n+1)
    for i in range(1,n+1):
        P[i]=P[i-1]+S[i-1]
    return P

def make_tuples(P,Q):
    T = list(zip(P,Q))
    return T

# def count_total(P, start, end):
#     total = P[end+1] - P[start]
#     return total
```

Solution with better performance.

```python
def solution(S, P, Q):
    res = []
    for i in range(len(P)):
        if 'A' in S[P[i]:Q[i]+1]:
            res.append(1)
        elif 'C' in S[P[i]:Q[i]+1]:
            res.append(2)
        elif 'G' in S[P[i]:Q[i]+1]:
            res.append(3)
        else:
            res.append(4)
    return res
```


### 2.14. MinAvgTwoSlice

Find starting index of the slice with the smallest avg.

Solution based on [Kadane's algorithm](https://en.wikipedia.org/wiki/Maximum_subarray_problem#Kadane%27s_algorithm). The basic question is (similar to that of the MSP):** What is the minimal average of a slice that includes the i-th element?**

Idea: Extend the best slice so far and compare it with the 2-element slice. If the extended best slice so far is better, extend it further. If the 2-element slice is better, make the new best slice so far.

```python
def solution(A):
    n = len(A)
    # calculate prefix sum
    p = [0]*(n+1)
    for i in range(1,n+1):
        p[i]=p[i-1]+A[i-1]

    avg_expand_best_slice_so_far = 0
    avg_two_slice = 0

    # -- Initialize with first slice --
    # initilize best slice (index)
    left_index = 0
    # keep track of best slice (index)
    min_left_index = left_index

    # initialize min average
    avg_here = (A[left_index]+A[left_index+1])/2
    # ikeep track of min average
    min_avg = avg_here

    # -- Find min avg of every slice that ends at i-th element, starting with the slice that ends at 2nd element; keep track of min average and best slice --
    for i in range(2,n):
        # expand best slice so far
        avg_expand_best_slice_so_far = (p[i+1]-p[left_index])/(i-left_index+1)
        # compute avg of two_slice
        avg_two_slice = (A[i-1]+A[i])/2

        # -- Question: which is better, the expanded best slice so far or the new two slice? --
        # two_slice is better
        if avg_two_slice < avg_expand_best_slice_so_far:
            avg_here = avg_two_slice
            # update new best slice (index)
            left_index = i-1

        # expanded best slice is better
        else:
            avg_here = avg_expand_best_slice_so_far

        # -- Keep track of min avg and best slice --
        if avg_here < min_avg:
            # keep track of min average
            min_avg = avg_here
            # keep track of best slice (index)
            min_left_index = left_index

    return min_left_index
```