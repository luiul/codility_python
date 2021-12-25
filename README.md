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
  - [2.15. Distinct](#215-distinct)
  - [2.16. MaxProductOfThree](#216-maxproductofthree)
  - [2.17. Triangle](#217-triangle)
  - [2.18. NumberOfDiscIntersections](#218-numberofdiscintersections)
  - [2.19. Brackets or Nesting](#219-brackets-or-nesting)
  - [2.20. Fish](#220-fish)
  - [2.21. StoneWall](#221-stonewall)
  - [2.22. Dominator](#222-dominator)
  - [2.23. EquiLeader](#223-equileader)
  - [2.24. FibFrog](#224-fibfrog)

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

    avg_extend_best_slice_so_far = 0
    avg_two_slice = 0

    # -- Initialize with first slice --
    # initilize best slice (index)
    left_index = 0
    # keep track of best slice (index)
    min_left_index = left_index

    # initialize min average
    avg_here = (A[left_index]+A[left_index+1])/2
    # keep track of min average
    min_avg = avg_here

    # -- Find min avg of every slice that ends at i-th element, starting with the slice that ends at 2nd element; keep track of min average and best slice --
    for i in range(2,n):
        # extend best slice so far
        avg_extend_best_slice_so_far = (p[i+1]-p[left_index])/(i-left_index+1)
        # compute avg of two_slice
        avg_two_slice = (A[i-1]+A[i])/2

        # -- Question: which is better, the extended best slice so far or the new two slice? --
        # two_slice is better
        if avg_two_slice < avg_extend_best_slice_so_far:
            avg_here = avg_two_slice
            # update new best slice (index)
            left_index = i-1

        # extended best slice is better
        else:
            avg_here = avg_extend_best_slice_so_far

        # -- Keep track of min avg and best slice --
        if avg_here < min_avg:
            # keep track of min average
            min_avg = avg_here
            # keep track of best slice (index)
            min_left_index = left_index

    return min_left_index
```


### 2.15. Distinct

Return number of distinct integers in a list.

```python
def solution(A):
    return len(set(A))
```

### 2.16. MaxProductOfThree

Maximize the multiplication of three elements in a list.

```python
def solution(A):
    A.sort(reverse=True)
    options = []
    # all elements are positive
    options.append(A[0]*A[1]*A[2])
    # at least one element is negative
    options.append(A[-1]*A[-2]*A[0])
    # all elements are negative
    options.append(A[-1]*A[-2]*A[-3])
    return max(options)
```

### 2.17. Triangle

Return 1 if there is a triangle in the list, otherwise return 0.

Naive solution:

```python
def solution(A):
    A.sort()
    for i in range(2,len(A)):
        if A[i] + A[i-1] > A[i-2] and A[i-1] + A[i-2] > A[i] and A[i-2] + A[i] > A[i-1]:
            return 1
    return 0
```

More compact solution:

```python
def solution(A):
    A.sort()
    for i in range(len(A)-2):
        if A[i] + A[i+1] > A[i+2]:
            return 1
    return 0
```

Explanation: if we sort the list, we can guarantee that A[i+2] + A[i+1] > A[i] and A[i+2] + A[i] > A[i+1] so we only have to check the third condition.

### 2.18. NumberOfDiscIntersections

Naive solution: Draw biggest circle and see if smaller circles overlap.

```python
def solution(A):
    n = len(A)
    i = list(range(n))
    tuples = list(zip(i,A))
    tuples.sort(key=lambda tup:tup[1], reverse=True)

    counter = 0

    # draw biggest circle
    for i, t in enumerate(tuples):
        c, r = t
        u = c+r
        l = c-r
        a = [l,u]
        # draw smallest circles
        for j in range(i+1,len(tuples)):
            sc, sr = tuples[j]
            su = sc + sr
            sl = sc - sr
            b = [sl,su]
            if get_overlap(a,b) > 0:
                counter += 1
    return counter

def get_overlap(x, y):
    return bool(range(max(x[0], y[0]), min(x[-1], y[-1])+1))
```

Solution with better performance: Computer lower and upper borders. Order them. Start with left upper border and count N lower borders left from it. N-1 are intersections. Move to the next upper border and repeat starting from the last lower border we considered.


```python
def solution(A):
    lower_list = []
    upper_list = []

    for center, radius in enumerate(A):
        lower_list.append(center - radius)
        upper_list.append(center + radius)

    lower_list.sort()
    upper_list.sort()

    j = 0
    counter = 0
    total_len = len(A)

    for i in range(total_len):
        while(j < total_len and lower_list[j] <= upper_list[i]):
            # if circle not inside circle: in the first iteration we check its own lower limit
            # if circle inside circle: in the last interation we check its own lower limit
            # -> in both cases it cancels each other out -> there is no intersection
            counter += j
            counter -= i
            # if next lower limit is left form the current upper limit -> there is an intersection
            j += 1

            if counter > 10000000:
                return -1

    return counter
```


### 2.19. Brackets or Nesting

Solution: Start with an empty stack. Append opening brackets. Pop opening brackets iff we see correct closing bracket, otherwise we know the string is not nested. Return the negation of the empty stack.

```python
def solution(S):
    # make dictionary with bracket pairs
    matches = dict(['()', '[]', '{}'])
    # make empty stack
    # an empty string is nested by default
    stack = []

    # iterate char of input string
    for char in S:
        # if char is oppening bracket, add it to the stack
        if char in matches.keys():
            stack.append(char)
        # if stack is not empty AND
        # last bracket in stack is key of char (bracket pair)
        # remove the last element
        # if this fails -> string is not properly nested
        if char in matches.values():
            if stack and matches[stack[-1]] == char:
                stack.pop()
            else:
                return 0

    # an empty string is nested by default
    return int(not stack)
```

### 2.20. Fish

Solution:

```python
def solution(A, B):
    size, direction = A, B
    # at the start all fishes are alive
    fish_alive = len(size)
    # empty list = no fishes alive
    if fish_alive == 0:
        return 0
    # make empty stack
    stack = []

    # iterate all fish numbers
    for fish_nr in range(fish_alive):
        # store fishes swimming downstream in a stack
        if direction[fish_nr] == 1:
            stack.append(size[fish_nr])

        # make fishes swimming upstream fight with stack of fishes swimming upstream
        if direction[fish_nr] == 0:
            # new fish has to fight all fishes swimming downstream
            while len(stack):
                # fish in stack is bigger -> new fish gets eaten
                if stack[-1] > size[fish_nr]:
                    fish_alive -= 1
                    break
                # fish in stack is smaller -> new fish eats stack fish and fights next fish in stack
                if stack[-1] < size[fish_nr]:
                    fish_alive -= 1
                    stack.pop()

    return fish_alive
```

### 2.21. StoneWall

Min the numbers of blocks required to build a wall given the height requirement list.

```python
def solution(H):
    req_height = H
    block_h = 0
    blocks = 0
    stack = []

    if len(req_height) in (0,1):
        return len(req_height)

    for i in range(len(req_height)):
        if not stack:
            stack.append(req_height[i])
            block_h = req_height[i] - 0
            blocks += 1

        if stack:
            while req_height[i] < sum(stack):
                stack.pop()
            if req_height[i] == sum(stack):
                pass
            if req_height[i] > sum(stack):
                delta_block = req_height[i] - sum(stack)
                stack.append(delta_block)
                blocks += 1

    return blocks
```

We remove the sum function for the stack and opt for keeping track of the current height with an additional variable.

```python
# remove the sum() statements to make it run faster
def solution(H):
    req_height = H
    blocks = 0
    stack = []
    current_height = 0

    if len(req_height) in (0,1):
        return len(req_height)

    for i in range(len(req_height)):
        if not stack:
            stack.append(req_height[i])
            blocks += 1
            current_height = req_height[i]

        if stack:
            while req_height[i] < current_height:
                current_height -= stack[-1]
                stack.pop()
            if req_height[i] == current_height:
                pass
            if req_height[i] > current_height:
                delta_block = req_height[i] - current_height
                stack.append(delta_block)
                blocks += 1
                current_height = req_height[i]

    return blocks
```

### 2.22. Dominator

Find an index of the list lead.

```python
def solution(A):
    n = len(A)
    if n == 0:
        return -1

    med = (n//2)
    lead = sorted(A)[med]
    count = 0

    for i in range(n):
        if lead == A[i]:
            count += 1
        if count > med:
            return i

    return -1
```

### 2.23. EquiLeader

Naive solution:

```python
def solution(A):
    n = len(A)
    # edge case
    if n in (0,1):
        return 0

    med = n//2
    lead = sorted(A)[med]
    count = 0

    for i in range(1,n):
        if A[:i:].count(lead) > len(A[:i:])//2 and A[i::].count(lead) > len(A[i::])//2:
            count += 1

    return count
```

Solution with better performance: Use a `defaultdict` and store {list_value : frequency} in it. Iterate through the keys of the dict and check for the equi-leader condition.

```python
from collections import defaultdict

def solution(A):
    # make two dictionaries
    # if they key does not exist, assign it to zero
    marker_l = defaultdict(lambda : 0)
    marker_r = defaultdict(lambda : 0)

    # right dict has (key=list_value : value=frequency)
    for i in range(len(A)):
        # in the first iteration we have 0 (default value in dict) + 1 = 1
        marker_r[A[i]] += 1

    # instatiate counter and default leader
    n_equi_leader = 0
    leader = A[0]


    # print(marker_r)
    # print(marker_l)
    # print('\n')
    #
    for i in range(len(A)):
        # reduce frequency of first list value in right marker
        marker_r[A[i]] -= 1
        # print(marker_r)
        # increase frequency of first list value in left marker (empty so far)
        marker_l[A[i]] += 1
        # print(marker_l)
        print('\n')

        # assign new leader if neccesary based on frequency
        if marker_l[leader] < marker_l[A[i]]:
            # assign new leader
            leader = A[i]

        # equi leader condition
        # the left dict is growing while the right dict is shrinking
        if (i+1) // 2 < marker_l[leader] and (len(A) - (i+1)) // 2 < marker_r[leader]:
            n_equi_leader += 1

    return n_equi_leader
```

Without the print statements:

```python
from collections import defaultdict

def solution(A):
    # make two dictionaries
    # if they key does not exist, assign it to zero
    marker_l = defaultdict(lambda : 0)
    marker_r = defaultdict(lambda : 0)

    # right dict has (key=list_value : value=frequency)
    for i in range(len(A)):
        # in the first iteration we have 0 (default value in dict) + 1 = 1
        marker_r[A[i]] += 1

    # instatiate counter and default leader
    n_equi_leader = 0
    leader = A[0]

    for i in range(len(A)):
        # reduce frequency of first list value in right marker
        marker_r[A[i]] -= 1
        # increase frequency of first list value in left marker (empty so far)
        marker_l[A[i]] += 1

        # assign new leader if neccesary based on frequency
        if marker_l[leader] < marker_l[A[i]]:
            # assign new leader
            leader = A[i]

        # equi leader condition
        # the left dict is growing while the right dict is shrinking
        if (i+1) // 2 < marker_l[leader] and (len(A) - (i+1)) // 2 < marker_r[leader]:
            n_equi_leader += 1

    return n_equi_leader
```

### 2.24. FibFrog

First naive solution is wrong. Idea: make the farthest jump possible first, make smaller jumps afterwards. After you make the farthest jump possible consider the rest of the list, starting from the current position.

```python
def solution(A):
    # get length of list
    n = len(A)
    if n < 3:
        return 1

    # make relevant fibs
    fib = [0,1]
    i = 1
    # relevant fibs
    while fib[i] < n:
        fib.append(fib[i] + fib[i-1])
        i += 1

    # relevant leaves
    B = [0]*(len(A)+1)
    for i in range(len(A)):
        if A[i]:
            B[i] = i+1
    B[i+1]=len(A)+1
    B = [i for i in B if i!=0]
    B.sort(reverse=True)

    # make possible jumps, default res is -1
    i = 0
    res = -1

    while B:
        if i == len(B):
            break
        if B[i] not in fib:
            pass
        if B[i] in fib:
            res += 1
            C = [x-B[i] for x in B]
            B = [i for i in C if i>0]
            # print(B)
            i = 0
            continue
        i += 1

    return res + 1
```

Similar idea to the naive solution but with better implementation and performance.

```python
def fib(n):
    fib = [0,1]
    i = 1
    # relevant fibs
    while fib[i] < n:
        fib.append(fib[i] + fib[i-1])
        i += 1
    return fib

def new_paths(A, n, last_pos, fn):
    paths = []
    # iterate the fib list
    for f in fn:
        # first iteration last_post = [-1]
        new_pos = last_pos + f
        # if news position is n or
        # new position in A is one
        # append the new leaf to the path
        if new_pos == n or (new_pos < n and A[new_pos]==1):
            paths.append(new_pos)
    return paths


def solution(A):
    # get length of list
    n = len(A)
    # edge case
    if n < 3:
        return 1

    # make fib list
    fn = fib(n)[2:]

    # starting position is [-1]
    paths = set([-1]))

    # initialize counter
    jump = 1

    while True:
        # Considering each of the previous jump positions - How many leaves from there are one fib jump away
        paths = set([idx for pos in paths for idx in new_paths(A, n, pos, fn)])

        # no new jumps means game over!
        if not paths:
            break

        # Return the number of jumps if [n] is in the path
        if n in paths:
            return jump

        # If paths is not empty record the jump
        jump += 1

    return -1
```