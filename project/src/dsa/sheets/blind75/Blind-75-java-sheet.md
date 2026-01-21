# Blind 75 Problems - Organized by Category

### Array Problems (15 problems)
1. Two Sum
2. Best Time to Buy and Sell Stock
3. Contains Duplicate
4. Product of Array Except Self
5. Maximum Subarray
6. Maximum Product Subarray
7. Find Minimum in Rotated Sorted Array
8. Search in Rotated Sorted Array
9. 3Sum
10. Container With Most Water
11. Set Matrix Zeroes
12. Spiral Matrix
13. Rotate Image
14. Word Search
15. Merge Intervals

### String Problems (9 problems)
16. Longest Substring Without Repeating Characters
17. Longest Repeating Character Replacement
18. Minimum Window Substring
19. Valid Anagram
20. Group Anagrams
21. Valid Parentheses
22. Valid Palindrome
23. Longest Palindromic Substring
24. Palindromic Substrings

### Tree Problems (11 problems)
25. Maximum Depth of Binary Tree
26. Same Tree
27. Invert Binary Tree
28. Binary Tree Maximum Path Sum
29. Binary Tree Level Order Traversal
30. Serialize and Deserialize Binary Tree
31. Subtree of Another Tree
32. Construct Binary Tree from Preorder and Inorder Traversal
33. Validate Binary Search Tree
34. Kth Smallest Element in a BST
35. Lowest Common Ancestor of a Binary Search Tree

### Linked List Problems (6 problems)
36. Reverse Linked List
37. Detect Cycle in Linked List
38. Merge Two Sorted Lists
39. Merge k Sorted Lists
40. Remove Nth Node From End of List
41. Reorder List

### Dynamic Programming Problems (11 problems)
42. Climbing Stairs
43. Coin Change
44. Longest Increasing Subsequence
45. Longest Common Subsequence
46. Word Break Problem
47. Combination Sum
48. House Robber
49. House Robber II
50. Decode Ways
51. Unique Paths
52. Jump Game

### Graph Problems (6 problems)
53. Clone Graph
54. Course Schedule
55. Pacific Atlantic Water Flow
56. Number of Islands
57. Longest Consecutive Sequence
58. Graph Valid Tree

### Binary/Bit Manipulation Problems (5 problems)
59. Sum of Two Integers
60. Number of 1 Bits
61. Counting Bits
62. Missing Number
63. Reverse Bits

### Heap Problems (3 problems)
64. Merge k Sorted Lists (also in Linked List)
65. Top K Frequent Elements
66. Find Median from Data Stream

### Interval Problems (4 problems)
67. Insert Interval
68. Merge Intervals (also in Array)
69. Non-overlapping Intervals
70. Meeting Rooms

### Trie Problems (3 problems)
71. Implement Trie (Prefix Tree)
72. Add and Search Word
73. Word Search II

### Advanced Problems (2 problems)
74. Alien Dictionary
75. Encode and Decode Strings

---

Now let me solve all the **Array Problems** first with optimal approaches:

## Array Problems Solutions

### 1. Two Sum

```java
import java.util.*;

/**
 * Problem: Given an array of integers nums and an integer target, 
 * return indices of the two numbers such that they add up to target.
 * 
 * Optimal Approach: HashMap for O(n) time complexity
 * Time: O(n), Space: O(n)
 */
public class TwoSum {
    public int[] twoSum(int[] nums, int target) {
        // HashMap to store value -> index mapping
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            
            // Check if complement exists in map
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            
            // Store current number and its index
            map.put(nums[i], i);
        }
        
        // No solution found (shouldn't happen as per problem guarantee)
        return new int[]{};
    }
}

/*
Key Insights:
1. Brute force O(n²) - check all pairs
2. Optimal O(n) - use HashMap to
 store complements
3. We iterate once, checking if complement exists
4. HashMap lookup is O(1) average case

Example walkthrough:
nums = [2,7,11,15], target = 9
i=0: complement = 9-2 = 7, map is empty, add {2:0}
i=1: complement = 9-7 = 2, found 2 at index 0, return [0,1]
*/
```

### 2. Best Time to Buy and Sell Stock

```java
/**
 * Problem: Find maximum profit from buying and selling stock once
 * 
 * Optimal Approach: Single pass tracking minimum price
 * Time: O(n), Space: O(1)
 */
public class BestTimeToBuyAndSellStock {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }
        
        int minPrice = prices[0];  // Track minimum price seen so far
        int maxProfit = 0;        // Track maximum profit possible
        
        for (int i = 1; i < prices.length; i++) {
            // Update minimum price if current price is lower
            minPrice = Math.min(minPrice, prices[i]);
            
            // Calculate profit if we sell at current price
            int currentProfit = prices[i] - minPrice;
            
            // Update maximum profit
            maxProfit = Math.max(maxProfit, currentProfit);
        }
        
        return maxProfit;
    }
}

/*
Key Insights:
1. We can only buy before we sell
2. To maximize profit, buy at lowest price before selling
3. Track minimum price seen so far
4. At each day, calculate profit if we sell today
5. Keep track of maximum profit seen

Example: [7,1,5,3,6,4]
Day 0: price=7, minPrice=7, profit=0
Day 1: price=1, minPrice=1, profit=0 (1-1=0)
Day 2: price=5, minPrice=1, profit=4 (5-1=4)
Day 3: price=3, minPrice=1, profit=4 (3-1=2, but max remains 4)
Day 4: price=6, minPrice=1, profit=5 (6-1=5)
Day 5: price=4, minPrice=1, profit=5 (4-1=3, but max remains 5)
*/
```

### 3. Contains Duplicate

```java
import java.util.*;

/**
 * Problem: Check if array contains any duplicates
 * 
 * Multiple approaches with different trade-offs
 */
public class ContainsDuplicate {
    
    // Approach 1: HashSet - Optimal for most cases
    // Time: O(n), Space: O(n)
    public boolean containsDuplicate1(int[] nums) {
        Set<Integer> seen = new HashSet<>();
        
        for (int num : nums) {
            // If number already exists in set, we found a duplicate
            if (seen.contains(num)) {
                return true;
            }
            seen.add(num);
        }
        
        return false;
    }
    
    // Approach 2: Sorting - Better space complexity
    // Time: O(n log n), Space: O(1) if in-place sort allowed
    public boolean containsDuplicate2(int[] nums) {
        Arrays.sort(nums);
        
        // Check adjacent elements after sorting
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                return true;
            }
        }
        
        return false;
    }
    
    // Approach 3: One-liner using Stream (less efficient but concise)
    public boolean containsDuplicate3(int[] nums) {
        return Arrays.stream(nums).distinct().count() != nums.length;
    }
}

/*
Approach Comparison:
1. HashSet: Best for most cases, O(n) time but O(n) space
2. Sorting: Good when space is limited, O(n log n) time but O(1) space
3. Stream: Readable but creates overhead

Choose based on constraints:
- Need fastest? Use HashSet
- Limited memory? Use sorting
- Code readability? Use stream (but be aware of performance cost)
*/
```

### 4. Product of Array Except Self

```java
/**
 * Problem: Return array where output[i] = product of all elements except nums
[i]
 * Constraint: Cannot use division operator
 * 
 * Optimal Approach: Two-pass with left and right products
 * Time: O(n), Space: O(1) excluding output array
 */
public class ProductOfArrayExceptSelf {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        
        // First pass: Calculate left products
        // result[i] = product of all elements to the left of i
        result[0] = 1;  // No elements to the left of index 0
        for (int i = 1; i < n; i++) {
            result[i] = result[i - 1] * nums[i - 1];
        }
        
        // Second pass: Calculate right products and multiply with left products
        int rightProduct = 1;  // Product of elements to the right
        for (int i = n - 1; i >= 0; i--) {
            result[i] *= rightProduct;  // Multiply left product with right product
            rightProduct *= nums[i];    // Update right product for next iteration
        }
        
        return result;
    }
}

/*
Algorithm Explanation:
Input: [1,2,3,4]

Step 1 - Left products:
result[0] = 1           (no elements to left)
result[1] = 1           (product of [1])
result[2] = 1*2 = 2     (product of [1,2])
result[3] = 2*3 = 6     (product of [1,2,3])
result = [1,1,2,6]

Step 2 - Right products:
i=3: result[3] = 6*1 = 6,     rightProduct = 1*4 = 4
i=2: result[2] = 2*4 = 8,     rightProduct = 4*3 = 12
i=1: result[1] = 1*12 = 12,   rightProduct = 12*2 = 24
i=0: result[0] = 1*24 = 24,   rightProduct = 24*1 = 24

Final result = [24,12,8,6]

Key Insights:
1. For each index i, we need: (product of left elements) × (product of right elements)
2. We can't use division, so we calculate left and right products separately
3. Use result array to store left products, then multiply with right products
4. Space complexity is O(1) if we don't count the output array
*/
```

### 5. Maximum Subarray (Kadane's Algorithm)

```java
/**
 * Problem: Find contiguous subarray with largest sum
 * 
 * Optimal Approach: Kadane's Algorithm
 * Time: O(n), Space: O(1)
 */
public class MaximumSubarray {
    public int maxSubArray(int[] nums) {
        // Initialize with first element
        int maxSoFar = nums[0];     // Global maximum
        int maxEndingHere = nums[0]; // Maximum sum ending at current position
        
        for (int i = 1; i < nums.length; i++) {
            // Either extend existing subarray or start new subarray
            maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
            
            // Update global maximum
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
    
    // Alternative implementation with clearer variable names
    public int maxSubArrayAlternative(int[] nums) {
        int globalMax = nums[0];
        int currentSum = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            // If current sum becomes negative, start fresh from current element
            currentSum = Math.max(nums[i], currentSum + nums[i]);
            globalMax = Math.max(globalMax, currentSum);
        }
        
        return globalMax;
    }
}

/*
Kadane's Algorithm Intuition:
1. At each position, decide: extend previous subarray OR start new subarray
2. Extend if adding current element increases the sum
3. Start new if current element alone is better than extended subarray
4. Keep track of maximum sum seen so far

Example: [-2,1,-3,4,-1,2,1,-5,4]
i=0: currentSum=-
2, globalMax=-2
i=1: currentSum=max(1, -2+1)=1, globalMax=max(-2,1)=1
i=2: currentSum=max(-3, 1-3)=-2, globalMax=max(1,-2)=1
i=3: currentSum=max(4, -2+4)=4, globalMax=max(1,4)=4
i=4: currentSum=max(-1, 4-1)=3, globalMax=max(4,3)=4
i=5: currentSum=max(2, 3+2)=5, globalMax=max(4,5)=5
i=6: currentSum=max(1, 5+1)=6, globalMax=max(5,6)=6
i=7: currentSum=max(-5, 6-5)=1, globalMax=max(6,1)=6
i=8: currentSum=max(4, 1+4)=5, globalMax=max(6,5)=6

The maximum subarray is [4,-1,2,1] with sum 6.

Why it works:
- If currentSum becomes negative, it's better to start fresh
- We're always looking for the best possible sum ending at current position
- Global maximum tracks the best sum we've seen across all positions
*/
```

### 6. Maximum Product Subarray

```java
/**
 * Problem: Find contiguous subarray with largest product
 * 
 * Optimal Approach: Track both max and min products (handle negative numbers)
 * Time: O(n), Space: O(1)
 */
public class MaximumProductSubarray {
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        
        int maxSoFar = nums[0];     // Global maximum product
        int maxEndingHere = nums[0]; // Max product ending at current position
        int minEndingHere = nums[0]; // Min product ending at current position
        
        for (int i = 1; i < nums.length; i++) {
            int current = nums[i];
            
            // Store maxEndingHere before updating (needed for minEndingHere calculation)
            int tempMax = maxEndingHere;
            
            // Calculate new max and min products ending at current position
            maxEndingHere = Math.max(current, Math.max(maxEndingHere * current, minEndingHere * current));
            minEndingHere = Math.min(current, Math.min(tempMax * current, minEndingHere * current));
            
            // Update global maximum
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
    
    // Alternative cleaner implementation
    public int maxProductAlternative(int[] nums) {
        int maxSoFar = nums[0];
        int maxHere = nums[0];
        int minHere = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            // When current number is negative, max and min swap roles
            if (nums[i] < 0) {
                int temp = maxHere;
                maxHere = minHere;
                minHere = temp;
            }
            
            // Update max and min products ending at current position
            maxHere = Math.max(nums[i], maxHere * nums[i]);
            minHere = Math.min(nums[i], minHere * nums[i]);
            
            // Update global maximum
            maxSoFar = Math.max(maxSoFar, maxHere);
        }
        
        return maxSoFar;
    }
}

/*
Key Insights:
1. Unlike sum, product can become very large or very small
2. Negative numbers can turn small products into large ones
3. We need to track both maximum and minimum products
4. When we encounter a negative number, max and min swap roles

Example: [2,3,-2,4]
i=0: max=2, min=2, global=2
i=1: max=max(3, max(2*3, 2*3))=6, min=min(3, min(2*3, 2*3))=3, global=6
i=2: current=-2 (negative), swap: max=3, min=6
      max=max(-2, max(3*-2, 6*-2))=max(-2, max(-6, -12))=-2
      min=min(-2, min(3*-2, 6*-2))=min(-2, min(-6, -12))=-12
      global=max(6, -2)=6
i=3: max=max(4, max(-2*4, -12*4))=max(4, max(-8, -48))=4
     min=min(4, min(-2*4, -12*4))=min(4, min(-8, -48))=-48
     global=max(6, 4)=6

The maximum product subarray is [2,3] with product 6.

Why track both max and min?
- A negative number can turn the smallest (most
 negative) product into the largest
- We need both to handle the sign changes correctly
*/
```

### 7. Find Minimum in Rotated Sorted Array

```java
/**
 * Problem: Find minimum element in rotated sorted array (no duplicates)
 * 
 * Optimal Approach: Binary Search
 * Time: O(log n), Space: O(1)
 */
public class FindMinimumInRotatedSortedArray {
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            // If mid element is greater than rightmost element,
            // minimum is in the right half
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } 
            // Otherwise, minimum is in left half (including mid)
            else {
                right = mid;
            }
        }
        
        return nums[left];
    }
    
    // Alternative approach comparing with left boundary
    public int findMinAlternative(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        
        // If array is not rotated
        if (nums[left] <= nums[right]) {
            return nums[left];
        }
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] >= nums[left]) {
                // Left half is sorted, minimum is in right half
                left = mid + 1;
            } else {
                // Right half is sorted, minimum is in left half
                right = mid;
            }
        }
        
        return nums[left];
    }
}

/*
Algorithm Explanation:
1. In a rotated sorted array, one half is always sorted
2. Compare mid with right boundary to determine which half contains minimum
3. If nums[mid] > nums[right]: minimum is in right half
4. Otherwise: minimum is in left half (including mid)

Example: [4,5,6,7,0,1,2]
left=0, right=6, mid=3, nums[3]=7, nums[6]=2
7 > 2, so minimum is in right half: left=4

left=4, right=6, mid=5, nums[5]=1, nums[6]=2  
1 < 2, so minimum is in left half: right=5

left=4, right=5, mid=4, nums[4]=0, nums[5]=1
0 < 1, so minimum is in left half: right=4

left=4, right=4, loop ends, return nums[4]=0

Why compare with right instead of left?
- Comparing with right helps us identify the rotation point
- The rotation point is where the minimum element is located
- If nums[mid] > nums[right], we know the rotation happened and minimum is to the right
- If nums[mid] <= nums[right], the minimum could be mid itself or to the left

Edge cases:
- Array not rotated: [1,2,3,4,5] -> return first element
- Single element: [1] -> return that element
- Two elements: [2,1] -> binary search will work correctly
*/
```

### 8. Search in Rotated Sorted Array

```java
/**
 * Problem: Search target in rotated sorted array
 * 
 * Optimal Approach: Modified Binary Search
 * Time: O(log n), Space: O(1)
 */
public class SearchInRotatedSortedArray {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            }
            
            // Determine which half is sorted
            if (nums[left] <= nums[mid]) {
                // Left half is sorted
                if (target >= nums[left] && target < nums[mid]) {
                    // Target is in sorted left half
                    right = mid - 1;
                } else {
                    // Target is in right half
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (target > nums[mid] && target <= nums[right]) {
                    // Target is in sorted right half
                    left = mid + 1;
                } else {
                    // Target is in left half
                    right = mid - 1;
                }
            }
        }
        
        return -1; // Target not found
    }
}

/*
Key Insights:
1. In rotated array, at least one half is always sorted
2. Identify which half is sorted by comparing nums[left] with nums[mid]
3. Check if target lies in the sorted half
4. If yes, search in sorted half; otherwise search in other half

Example: [4,5,6,7,0,1,2],
 target = 0
left=0, right=6, mid=3, nums[3]=7
nums[0]=4 <= nums[3]=7, so left half [4,5,6,7] is sorted
target=0 not in [4,7), so search right half: left=4

left=4, right=6, mid=5, nums[5]=1  
nums[4]=0 > nums[5]=1, so right half [1,2] is sorted
target=0 not in (1,2], so search left half: right=4

left=4, right=4, mid=4, nums[4]=0
Found target at index 4!

Algorithm breakdown:
1. Check if mid is target
2. Determine which half is sorted:
   - If nums[left] <= nums[mid]: left half is sorted
   - Otherwise: right half is sorted
3. Check if target is in the sorted half:
   - If yes: search in sorted half
   - If no: search in unsorted half
4. Repeat until found or search space exhausted

Why this works:
- At least one half is always sorted in a rotated array
- We can use normal binary search logic on the sorted half
- For the unsorted half, we recursively apply the same logic
*/
```

### 9. 3Sum

```java
import java.util.*;

/**
 * Problem: Find all unique triplets that sum to zero
 * 
 * Optimal Approach: Sort + Two Pointers
 * Time: O(n²), Space: O(1) excluding output
 */
public class ThreeSum {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        
        if (nums == null || nums.length < 3) {
            return result;
        }
        
        // Sort the array to enable two-pointer technique
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 2; i++) {
            // Skip duplicates for first element
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            
            int left = i + 1;
            int right = nums.length - 1;
            int target = -nums[i]; // We want nums[left] + nums[right] = target
            
            while (left < right) {
                int sum = nums[left] + nums[right];
                
                if (sum == target) {
                    // Found a triplet
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    
                    // Skip duplicates for second element
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    // Skip duplicates for third element
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    
                    left++;
                    right--;
                } else if (sum < target) {
                    left++; // Need larger sum
                } else {
                    right--; // Need smaller sum
                }
            }
        }
        
        return result;
    }
}

/*
Algorithm Steps:
1. Sort array to enable two-pointer technique
2. Fix first element, use two pointers for remaining two
3. Skip duplicates to avoid duplicate triplets
4. Use two-pointer technique to find pairs that sum to -nums[i]

Example: [-1,0,1,2,-1,-4]
After sorting: [-4,-1,-1,0,1,2]

i=0, nums[0]=-4, target=4
left=1(-1), right=5(2), sum=-1+2=1 < 4, left++
left=2(-1), right=5(2), sum=-1+2=1 < 4, left++  
left=3(0), right=5(2), sum=0+2=2 < 4, left++
left=4(1), right=5(2), sum=1+2=3 < 4, left++
left=5, right=5, left >= right, exit

i=1, nums[1]=-1, target=1
left=2(-1), right=5(2), sum=-1+2=1 = 1, found triplet [-1,-1,2]
Skip duplicates: left=3, right=4
left=3(0), right=4(1), sum=0+1=1 = 1, found triplet [-1,0,1]

i=2, nums[2]=-1, skip duplicate (same as nums[1])

i=3, nums[3]=0, target=0
left=4(1), right=5(2), sum=1+2=3 > 0, right--
left=4, right=4, left >= right, exit

Result: [[-1,-1,2], [-1,0,1]]

Time Complexity: O(n²) - O(n log n) for sorting + O(n²) for neste
d loops
Space Complexity: O(1) if we don't count output space

Key optimizations:
1. Sort first to enable two-pointer technique
2. Skip duplicates to avoid duplicate triplets
3. Use two-pointer technique instead of nested loops for inner search
*/
```

### 10. Container With Most Water

```java
/**
 * Problem: Find two lines that form container with most water
 * 
 * Optimal Approach: Two Pointers
 * Time: O(n), Space: O(1)
 */
public class ContainerWithMostWater {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            // Calculate current area
            int width = right - left;
            int currentHeight = Math.min(height[left], height[right]);
            int currentArea = width * currentHeight;
            
            // Update maximum area
            maxArea = Math.max(maxArea, currentArea);
            
            // Move the pointer with smaller height
            // This gives us the best chance to find a larger area
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
    }
}

/*
Why move the pointer with smaller height?
1. Area is limited by the shorter line
2. Moving the taller line won't increase area (width decreases, height stays same)
3. Moving the shorter line might find a taller line, potentially increasing area
4. This greedy approach ensures we don't miss the optimal solution

Example: [1,8,6,2,5,4,8,3,7]
left=0(1), right=8(7), area = 8 * min(1,7) = 8 * 1 = 8
Move left (smaller height): left=1

left=1(8), right=8(7), area = 7 * min(8,7) = 7 * 7 = 49
Move right (smaller height): right=7

left=1(8), right=7(3), area = 6 * min(8,3) = 6 * 3 = 18
Move right (smaller height): right=6

left=1(8), right=6(8), area = 5 * min(8,8) = 5 * 8 = 40
Either can move, let's move left: left=2

Continue until left >= right...
Maximum area found is 49.

Proof of correctness:
- We start with maximum width
- At each step, we eliminate one line that cannot be part of optimal solution
- The line we eliminate is the shorter one, because keeping it with any other line will give smaller area than what we already calculated
- This greedy approach guarantees we find the optimal solution

Intuition:
- Start with widest possible container
- The area is always limited by the shorter line
- To potentially get a larger area, we must find a taller line
- Moving the taller line is pointless (width decreases, height can't increase)
- Moving the shorter line gives us a chance to find a taller line
*/
```

### 11. Set Matrix Zeroes

```java
/**
 * Problem: Set entire row and column to zero if element is zero
 * 
 * Optimal Approach: Use first row and column as markers
 * Time: O(m*n), Space: O(1)
 */
public class SetMatrixZeroes {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        
        boolean firstRowZero = false;
        boolean firstColZero = false;
        
        // Check if first row should be zero
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                firstRowZero = true;
                break;
            }
        }
        
        // Check if first column should be zero
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                firstColZero = true;
                break;
            }
        }
        
        // Use first row and column as markers
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0; // Mark row
                    matrix[0][j] = 0; // Mark column
                }
            }
        }
        
        // Set zeros based on markers (excluding first row and column)
        for (int i = 1; i
 < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // Handle first row
        if (firstRowZero) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
        
        // Handle first column
        if (firstColZero) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }
    
    // Alternative approach using extra space
    // Time: O(m*n), Space: O(m+n)
    public void setZeroesWithExtraSpace(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        
        Set<Integer> zeroRows = new HashSet<>();
        Set<Integer> zeroCols = new HashSet<>();
        
        // Find all zero positions
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    zeroRows.add(i);
                    zeroCols.add(j);
                }
            }
        }
        
        // Set rows to zero
        for (int row : zeroRows) {
            for (int j = 0; j < n; j++) {
                matrix[row][j] = 0;
            }
        }
        
        // Set columns to zero
        for (int col : zeroCols) {
            for (int i = 0; i < m; i++) {
                matrix[i][col] = 0;
            }
        }
    }
}

/*
Algorithm Explanation (Optimal O(1) space):

1. Use first row and first column as markers
2. Before using them as markers, check if they originally contain zeros
3. Scan the matrix and mark corresponding first row/column positions
4. Use the markers to set zeros in the rest of the matrix
5. Finally handle the first row and column based on original state

Example: 
[[1,1,1],
 [1,0,1],
 [1,1,1]]

Step 1: Check first row and column
firstRowZero = false, firstColZero = false

Step 2: Mark zeros
matrix[1][1] = 0, so mark matrix[1][0] = 0 and matrix[0][1] = 0
Matrix becomes:
[[1,0,1],
 [0,0,1],
 [1,1,1]]

Step 3: Set zeros based on markers
matrix[1][0] = 0, so set entire row 1 to zero
matrix[0][1] = 0, so set entire column 1 to zero
Matrix becomes:
[[1,0,1],
 [0,0,0],
 [1,0,1]]

Step 4: Handle first row and column (no changes needed)

Why this works:
- We use the matrix itself to store information about which rows/columns to zero
- First row and column serve as our "memory" for which rows/columns had zeros
- We handle the first row and column separately to avoid conflicts

Space complexity analysis:
- O(m+n) approach: Use separate sets to track zero rows and columns
- O(1) approach: Use the matrix itself as storage, only need a few boolean variables
*/
```

### 12. Spiral Matrix

```java
import java.util.*;

/**
 * Problem: Return elements of matrix in spiral order
 * 
 * Optimal Approach: Layer by layer traversal
 * Time: O(m*n), Space: O(1) excluding output
 */
public class SpiralMatrix {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }
        
        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        
        while (top <= bottom && left <= right) {
            // Traverse right along top row
            for (int j = left; j <= right; j++) {
                result.add(matrix[top][j]);
            }
            top++;
            
            // Traverse down along right column
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;
            
            // Traverse left along bottom row (if we still have rows)
            if (top <= bottom) {
                for (int j = right; j >= left; j--) {
                    result.add(matrix[bottom][j]);
                }
                bottom--;
            }
            
            // Traverse up along left column (if we still have columns)
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
        }
        
        return result;
    }
    
    // Alternative approach using direction
 vectors
    public List<Integer> spiralOrderDirectional(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        
        if (matrix == null || matrix.length == 0) return result;
        
        int m = matrix.length, n = matrix[0].length;
        boolean[][] visited = new boolean[m][n];
        
        // Direction vectors: right, down, left, up
        int[] dr = {0, 1, 0, -1};
        int[] dc = {1, 0, -1, 0};
        
        int r = 0, c = 0, di = 0; // Start at (0,0) going right
        
        for (int i = 0; i < m * n; i++) {
            result.add(matrix[r][c]);
            visited[r][c] = true;
            
            // Calculate next position
            int nr = r + dr[di];
            int nc = c + dc[di];
            
            // Check if we need to turn (hit boundary or visited cell)
            if (nr < 0 || nr >= m || nc < 0 || nc >= n || visited[nr][nc]) {
                di = (di + 1) % 4; // Turn clockwise
                nr = r + dr[di];
                nc = c + dc[di];
            }
            
            r = nr;
            c = nc;
        }
        
        return result;
    }
}

/*
Algorithm Explanation (Layer approach):

We traverse the matrix layer by layer from outside to inside.
For each layer, we traverse in this order:
1. Top row: left to right
2. Right column: top to bottom  
3. Bottom row: right to left (if there are still rows)
4. Left column: bottom to top (if there are still columns)

Example: 
[[1,2,3],
 [4,5,6],
 [7,8,9]]

Layer 1 (outer):
- Top row: 1,2,3 (left=0, right=2, top=0)
- Right column: 6,9 (right=2, top=1, bottom=2)  
- Bottom row: 8,7 (bottom=2, right=1, left=0)
- Left column: 4 (left=0, bottom=1, top=1)

Layer 2 (inner):
- Only element 5 remains

Result: [1,2,3,6,9,8,7,4,5]

Key insights:
1. Use four boundaries: top, bottom, left, right
2. After each direction, update the corresponding boundary
3. Check if boundaries are still valid before traversing bottom and left
4. This handles edge cases like single row or single column matrices

Time Complexity: O(m*n) - visit each element once
Space Complexity: O(1) - only use a few variables (excluding output)

Edge cases handled:
- Single row: only traverse right, skip bottom and left
- Single column: traverse right and down, skip bottom and left  
- Single element: traverse right only
- Empty matrix: return empty list
*/
```

### 13. Rotate Image

```java
/**
 * Problem: Rotate n×n matrix 90 degrees clockwise in-place
 * 
 * Optimal Approach: Transpose + Reverse rows
 * Time: O(n²), Space: O(1)
 */
public class RotateImage {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        
        // Step 1: Transpose the matrix (swap matrix[i][j] with matrix[j][i])
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        // Step 2: Reverse each row
        for (int i = 0; i < n; i++) {
            reverseRow(matrix[i]);
        }
    }
    
    private void reverseRow(int[] row) {
        int left = 0;
        int right = row.length - 1;
        
        while (left < right) {
            int temp = row[left];
            row[left] = row[right];
            row[right] = temp;
            left++;
            right--;
        }
    }
    
    // Alternative: Direct rotation using 4-way swap
    public void rotateDirectly(int[][] matrix) {
        int n = matrix.length;
        
        // Process layer by layer
        for (int layer = 0; layer < n / 2; layer++) {
            int first
 = layer;
            int last = n - 1 - layer;
            
            for (int i = first; i < last; i++) {
                int offset = i - first;
                
                // Save top element
                int top = matrix[first][i];
                
                // Top = Left
                matrix[first][i] = matrix[last - offset][first];
                
                // Left = Bottom
                matrix[last - offset][first] = matrix[last][last - offset];
                
                // Bottom = Right
                matrix[last][last - offset] = matrix[i][last];
                
                // Right = Top (saved)
                matrix[i][last] = top;
            }
        }
    }
}

/*
Algorithm Explanation (Transpose + Reverse):

90-degree clockwise rotation can be achieved by:
1. Transpose the matrix (reflect along main diagonal)
2. Reverse each row (reflect along vertical center line)

Example:
Original:        Transpose:       Reverse rows:
[1,2,3]         [1,4,7]          [7,4,1]
[4,5,6]   -->   [2,5,8]   -->    [8,5,2]
[7,8,9]         [3,6,9]          [9,6,3]

Step-by-step:
1. Transpose: matrix[i][j] ↔ matrix[j][i]
   [1,2,3]     [1,4,7]
   [4,5,6] --> [2,5,8]
   [7,8,9]     [3,6,9]

2. Reverse each row:
   [1,4,7]     [7,4,1]
   [2,5,8] --> [8,5,2]
   [3,6,9]     [9,6,3]

Why this works:
- Transpose reflects the matrix along the main diagonal
- Reversing rows reflects along the vertical center
- Combined effect is a 90-degree clockwise rotation

Alternative approach (Direct 4-way swap):
For each layer, rotate 4 elements at a time:
- Save top element
- Move left to top
- Move bottom to left  
- Move right to bottom
- Move saved top to right

Time Complexity: O(n²) - must touch every element
Space Complexity: O(1) - only use temporary variables

For counter-clockwise rotation:
- Transpose + reverse columns, OR
- Reverse rows + transpose
*/
```

### 14. Word Search

```java
/**
 * Problem: Find if word exists in 2D board (can move in 4 directions)
 * 
 * Optimal Approach: DFS with backtracking
 * Time: O(m*n*4^L) where L is word length, Space: O(L) for recursion stack
 */
public class WordSearch {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || word == null || word.length() == 0) {
            return false;
        }
        
        int m = board.length;
        int n = board[0].length;
        
        // Try starting from each cell
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private boolean dfs(char[][] board, String word, int i, int j, int index) {
        // Base case: found the complete word
        if (index == word.length()) {
            return true;
        }
        
        // Check boundaries and character match
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
            board[i][j] != word.charAt(index)) {
            return false;
        }
        
        // Mark current cell as visited
        char temp = board[i][j];
        board[i][j] = '#'; // Use a marker to indicate visited
        
        // Explore all 4 directions
        boolean found = dfs(board, word, i + 1, j, index + 1) ||  // down
                       dfs(board, word, i - 1, j, index + 1) ||  // up
                       dfs(board, word, i, j + 1, index + 1) ||  // right
                       dfs(board, word, i, j - 1, index + 1);    // left
        
        // Backtrack: restore the original character
        board[i][j] = temp;
        
        return found;
    }
    
    // Alternative implementation with explicit visited array
    public boolean existWithVisited(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        
        for (int i = 0; i < m; i++) {
            for (int j = 0;
 j < n; j++) {
                if (dfsWithVisited(board, word, i, j, 0, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private boolean dfsWithVisited(char[][] board, String word, int i, int j, 
                                  int index, boolean[][] visited) {
        if (index == word.length()) return true;
        
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
            visited[i][j] || board[i][j] != word.charAt(index)) {
            return false;
        }
        
        visited[i][j] = true;
        
        boolean found = dfsWithVisited(board, word, i + 1, j, index + 1, visited) ||
                       dfsWithVisited(board, word, i - 1, j, index + 1, visited) ||
                       dfsWithVisited(board, word, i, j + 1, index + 1, visited) ||
                       dfsWithVisited(board, word, i, j - 1, index + 1, visited);
        
        visited[i][j] = false; // backtrack
        
        return found;
    }
}

/*
Algorithm Explanation:

1. Try starting the search from every cell in the board
2. For each starting position, use DFS to explore all possible paths
3. At each step, check if current character matches the expected character
4. Mark current cell as visited to avoid cycles
5. Recursively explore all 4 directions
6. Backtrack by unmarking the cell when returning from recursion

Example:
Board: [['A','B','C','E'],
        ['S','F','C','S'],
        ['A','D','E','E']]
Word: "ABCCED"

Starting from (0,0) 'A':
- Match 'A' at (0,0), mark as visited, look for 'B'
- Find 'B' at (0,1), mark as visited, look for 'C'  
- Find 'C' at (0,2), mark as visited, look for 'C'
- Find 'C' at (1,2), mark as visited, look for 'E'
- Find 'E' at (2,2), mark as visited, look for 'D'
- Find 'D' at (2,1), mark as visited, word complete!

Backtracking is crucial:
- When we mark a cell as visited, we prevent revisiting it in the current path
- When we backtrack, we unmark it so it can be used in other paths
- This ensures we explore all possible paths without getting stuck in cycles

Time Complexity: O(m*n*4^L)
- m*n: try starting from each cell
- 4^L: at each step, we have up to 4 directions, and word length is L

Space Complexity: O(L) 
- Recursion stack depth is at most the word length
- If using explicit visited array: O(m*n + L)

Optimizations:
1. Modify board in-place instead of using visited array (saves space)
2. Early termination if remaining characters can't possibly form the word
3. Check if all characters in word exist in board before starting search
*/
```

### 15. Merge Intervals

```java
import java.util.*;

/**
 * Problem: Merge overlapping intervals
 * 
 * Optimal Approach: Sort by start time + merge
 * Time: O(n log n), Space: O(1) excluding output
 */
public class MergeIntervals {
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length <= 1) {
            return intervals;
        }
        
        // Sort intervals by start time
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        
        List<int[]> merged = new ArrayList<>();
        
        for (int[] interval : intervals) {
            // If merged is empty or current interval doesn't overlap with the last one
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] < interval[0]) {
                merged.add(interval);
            } else {
                // Overlapping intervals, merge them
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], interval[1]);
            }
        }
        
        return merged.toArray(new int[merged.size()][]);
    }
    
    // Alternative implementation with clearer variable names
    public int[][] mergeAlternative(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        // Sort by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> result = new ArrayList<>();
        int[] currentInterval = intervals[0];
        result.add(currentInterval);
        
        for (int[] interval : intervals) {
            int currentEnd = current
Interval[1];
            int nextStart = interval[0];
            int nextEnd = interval[1];
            
            if (currentEnd >= nextStart) {
                // Overlapping intervals, merge them
                currentInterval[1] = Math.max(currentEnd, nextEnd);
            } else {
                // Non-overlapping interval, add to result
                currentInterval = interval;
                result.add(currentInterval);
            }
        }
        
        return result.toArray(new int[result.size()][]);
    }
}

/*
Algorithm Explanation:

1. Sort intervals by start time to process them in order
2. Iterate through sorted intervals
3. For each interval:
   - If it doesn't overlap with the last merged interval, add it to result
   - If it overlaps, merge by extending the end time of the last interval

Example: [[1,3],[2,6],[8,10],[15,18]]

After sorting: [[1,3],[2,6],[8,10],[15,18]] (already sorted)

Process:
1. Add [1,3] to result: [[1,3]]
2. [2,6]: 2 <= 3, so overlapping. Merge: [[1,6]]
3. [8,10]: 8 > 6, so non-overlapping. Add: [[1,6],[8,10]]
4. [15,18]: 15 > 10, so non-overlapping. Add: [[1,6],[8,10],[15,18]]

Overlap condition:
Two intervals [a,b] and [c,d] overlap if:
- a <= c <= b (c starts before b ends), OR
- c <= a <= d (a starts before d ends)

Since we sort by start time, we only need to check: lastEnd >= currentStart

Edge cases:
1. Empty array: return as is
2. Single interval: return as is  
3. No overlaps: return all intervals
4. All intervals overlap: return single merged interval
5. Identical intervals: merge into one

Time Complexity: O(n log n) due to sorting
Space Complexity: O(1) if we don't count output space, O(n) including output

Why sorting works:
- After sorting by start time, we only need to compare each interval with the previous one
- If current interval overlaps with previous, they must be merged
- If not, we can safely add current interval to result
- This greedy approach works because sorting ensures optimal order
*/
```

That completes all 15 Array problems! Each solution includes:

1. **Optimal approach** with best time/space complexity
2. **Detailed algorithm explanation** with step-by-step walkthrough
3. **Key insights** and intuition behind the solution
4. **Example demonstrations** showing how the algorithm works
5. **Alternative approaches** when relevant
6. **Edge case handling** and complexity analysis

-----------------------------------------------------------------



## String Problems Solutions

### 16. Longest Substring Without Repeating Characters

```java
import java.util.*;

/**
 * Problem: Find length of longest substring without repeating characters
 * 
 * Optimal Approach: Sliding Window with HashMap
 * Time: O(n), Space: O(min(m,n)) where m is charset size
 */
public class LongestSubstringWithoutRepeatingCharacters {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) return 0;
        
        Map<Character, Integer> charIndexMap = new HashMap<>();
        int maxLength = 0;
        int left = 0; // Left pointer of sliding window
        
        for (int right = 0; right < s.length(); right++) {
            char currentChar = s.charAt(right);
            
            // If character is already in current window, move left pointer
            if (charIndexMap.containsKey(currentChar)) {
                // Move left to position after the duplicate character
                left = Math.max(left, charIndexMap.get(currentChar) + 1);
            }
            
            // Update character's latest index
            charIndexMap.put(currentChar, right);
            
            // Update maximum length
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // Alternative: Using HashSet with explicit window shrinking
    public int lengthOfLongestSubstringSet(String s) {
        Set<Character> window = new HashSet<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char currentChar = s.charAt(right);
            
            // Shrink window until no duplicate
            while (window.contains(currentChar)) {
                window.remove(s.charAt(left));
                left++;
            }
            
            window.add(currentChar);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // For ASCII characters only - using array instead of HashMap
    public int lengthOfLongestSubstringArray(String s) {
        int[] charIndex = new int[128]; // ASCII characters
        Arrays.fill(charIndex, -1);
        
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char currentChar = s.charAt(right);
            
            if (charIndex[currentChar] >= left) {
                left = charIndex[currentChar] + 1;
            }
            
            charIndex[currentChar] = right;
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}

/*
Algorithm Explanation (Sliding Window):

We maintain a sliding window [left, right] that contains no duplicate characters.

1. Expand window by moving right pointer
2. If we encounter a duplicate character:
   - Move left pointer to position after the previous occurrence
3. Update maximum length at each step

Example: "abcabcbb"
right=0, char='a': window="a", maxLength=1
right=1, char='b': window="ab", maxLength=2  
right=2, char='c': window="abc", maxLength=3
right=3, char='a': 'a' seen at index 0, move left=1, window="bca", maxLength=3
right=4, char='b': 'b' seen at index 1, move left=2, window="cab", maxLength=3
right=5, char='c': 'c' seen at index 2, move left=3, window="abc", maxLength=3
right=6, char='b': 'b' seen at index 4, move left=5, window="cb", maxLength=3
right=7, char='b': 'b' seen at index 6, move left=7, window="b", maxLength=3

Key Insights:
1. Use HashMap to store character -> latest index mapping
2. When duplicate found, jump left pointer to avoid redundant checks
3. Always update character's index to current position
4. Window size = right - left + 1

Time Complexity: O(n) - each character visited at most twice

Space Complexity: O(min(m,n)) where m is charset size

Optimization for ASCII:
- Use array instead of HashMap for better performance
- Array access is faster than HashMap operations
- Only works if character set is limited (ASCII, Unicode, etc.)

Why this approach works:
- Sliding window ensures we always maintain valid substring
- HashMap allows O(1) lookup for duplicate detection
- Jumping left pointer avoids unnecessary character-by-character movement
*/
```

### 17. Longest Repeating Character Replacement

```java
import java.util.*;

/**
 * Problem: Find longest substring with same character after at most k replacements
 * 
 * Optimal Approach: Sliding Window with character frequency tracking
 * Time: O(n), Space: O(1) - at most 26 characters
 */
public class LongestRepeatingCharacterReplacement {
    public int characterReplacement(String s, int k) {
        if (s == null || s.length() == 0) return 0;
        
        int[] charCount = new int[26]; // Frequency of each character in current window
        int maxLength = 0;
        int maxFreq = 0; // Frequency of most frequent character in current window
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char rightChar = s.charAt(right);
            charCount[rightChar - 'A']++;
            
            // Update max frequency in current window
            maxFreq = Math.max(maxFreq, charCount[rightChar - 'A']);
            
            // Current window size
            int windowSize = right - left + 1;
            
            // If replacements needed > k, shrink window
            if (windowSize - maxFreq > k) {
                char leftChar = s.charAt(left);
                charCount[leftChar - 'A']--;
                left++;
                // Note: We don't update maxFreq here for optimization
                // It's okay to have stale maxFreq as it only affects window size calculation
                // but won't affect the final result
            }
            
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // Alternative: More intuitive but slightly less efficient
    public int characterReplacementIntuitive(String s, int k) {
        Map<Character, Integer> charCount = new HashMap<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char rightChar = s.charAt(right);
            charCount.put(rightChar, charCount.getOrDefault(rightChar, 0) + 1);
            
            // Find max frequency in current window
            int maxFreq = Collections.max(charCount.values());
            int windowSize = right - left + 1;
            
            // If we need more than k replacements, shrink window
            while (windowSize - maxFreq > k) {
                char leftChar = s.charAt(left);
                charCount.put(leftChar, charCount.get(leftChar) - 1);
                if (charCount.get(leftChar) == 0) {
                    charCount.remove(leftChar);
                }
                left++;
                windowSize = right - left + 1;
                if (!charCount.isEmpty()) {
                    maxFreq = Collections.max(charCount.values());
                }
            }
            
            maxLength = Math.max(maxLength, windowSize);
        }
        
        return maxLength;
    }
}

/*
Algorithm Explanation:

The key insight is that we want the longest substring where:
(substring length - frequency of most common character) <= k

This means we can replace at most k characters to make all characters the same.

Example: s = "AABABBA", k = 1

Window expansion:
right=0: "A", maxFreq=1, windowSize=1, replacements=0 ≤ 1 ✓
right=1: "AA", maxFreq=2, windowSize=2, replacements=0 ≤ 1 ✓  
right=2: "AAB", maxFreq=2, windowSize=3, replacements=1 ≤ 1 ✓
right=3: "AABA", maxFreq=3, windowSize=4, replacements=1 ≤ 1 ✓
right=4: "AABAB", maxFreq=3, windowSize=5, replacements=2 > 1 ✗

Window
 shrinking:
left=1: "ABAB", maxFreq=2, windowSize=4, replacements=2 > 1 ✗
left=2: "BAB", maxFreq=2, windowSize=3, replacements=1 ≤ 1 ✓

Continue...

Key Insights:
1. Use sliding window to maintain valid substring
2. Track frequency of each character in current window
3. Most frequent character should be kept, others replaced
4. If replacements needed > k, shrink window from left

Optimization in first solution:
- We don't update maxFreq when shrinking window
- This is safe because:
  - Stale maxFreq might make us think window is larger than it is
  - But we only care about finding the maximum valid window
  - If current window isn't optimal due to stale maxFreq, we'll find the optimal one later

Time Complexity: O(n) - each character processed at most twice
Space Complexity: O(1) - at most 26 characters in English alphabet

Why sliding window works:
- We maintain the invariant that current window needs at most k replacements
- When invariant is violated, we shrink from left until it's satisfied
- We track the maximum valid window size throughout the process
*/
```

### 18. Minimum Window Substring

```java
import java.util.*;

/**
 * Problem: Find minimum window in s that contains all characters of t
 * 
 * Optimal Approach: Sliding Window with character frequency matching
 * Time: O(|s| + |t|), Space: O(|s| + |t|)
 */
public class MinimumWindowSubstring {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() < t.length()) {
            return "";
        }
        
        // Count characters in t
        Map<Character, Integer> targetCount = new HashMap<>();
        for (char c : t.toCharArray()) {
            targetCount.put(c, targetCount.getOrDefault(c, 0) + 1);
        }
        
        int required = targetCount.size(); // Number of unique characters in t
        int formed = 0; // Number of unique characters in current window with desired frequency
        
        Map<Character, Integer> windowCount = new HashMap<>();
        
        int left = 0, right = 0;
        int minLength = Integer.MAX_VALUE;
        int minLeft = 0; // Starting index of minimum window
        
        while (right < s.length()) {
            // Expand window by including character at right
            char rightChar = s.charAt(right);
            windowCount.put(rightChar, windowCount.getOrDefault(rightChar, 0) + 1);
            
            // Check if frequency of current character matches target frequency
            if (targetCount.containsKey(rightChar) && 
                windowCount.get(rightChar).intValue() == targetCount.get(rightChar).intValue()) {
                formed++;
            }
            
            // Try to shrink window from left
            while (left <= right && formed == required) {
                // Update minimum window if current is smaller
                if (right - left + 1 < minLength) {
                    minLength = right - left + 1;
                    minLeft = left;
                }
                
                // Remove character at left from window
                char leftChar = s.charAt(left);
                windowCount.put(leftChar, windowCount.get(leftChar) - 1);
                
                if (targetCount.containsKey(leftChar) && 
                    windowCount.get(leftChar).intValue() < targetCount.get(leftChar).intValue()) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return minLength == Integer.MAX_VALUE ? "" : s.substring(minLeft, minLeft + minLength);
    }
    
    // Optimized version using arrays for ASCII characters
    public String minWindowOptimized(String s, String t) {
        if (s.length() < t.length()) return "";
        
        int[] targetCount = new int[128];
        int[] windowCount = new int[128];
        
        // Count characters in t
        for (char c : t.toCharArray()) {
            targetCount[c]++;
        }
        
        int required = 0;
        for (int count : targetCount) {
            if (count > 0) required++;
        }
        
        int formed = 0;
        int left = 0;
        int minLength = Integer.MAX_VALUE;
        int minLeft = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char rightChar = s.charAt(right);
            windowCount[rightChar]++;
            
            if (targetCount[rightChar] > 0 && windowCount[rightChar] == targetCount[rightChar]) {
                formed++;
            }
            
            while (formed == required) {
                if (right - left + 1 < minLength) {
                    minLength = right - left + 1;
                    minLeft = left;
                }
                
                char leftChar = s.charAt(left);
                windowCount[leftChar]--;
                
                if (targetCount[leftChar] > 0 && windowCount[leftChar] < targetCount[
leftChar]) {
                    formed--;
                }
                
                left++;
            }
        }
        
        return minLength == Integer.MAX_VALUE ? "" : s.substring(minLeft, minLeft + minLength);
    }
}

/*
Algorithm Explanation:

We use sliding window technique with two pointers:
1. Expand window by moving right pointer until all characters of t are included
2. Once valid window found, try to shrink from left while maintaining validity
3. Track the minimum valid window throughout the process

Example: s = "ADOBECODEBANC", t = "ABC"

Target: A=1, B=1, C=1 (required = 3)

Expansion phase:
right=0: A, window={A:1}, formed=1
right=1: AD, window={A:1,D:1}, formed=1
right=2: ADO, window={A:1,D:1,O:1}, formed=1
right=3: ADOB, window={A:1,D:1,O:1,B:1}, formed=2
right=4: ADOBE, window={A:1,D:1,O:1,B:1,E:1}, formed=2
right=5: ADOBEC, window={A:1,D:1,O:1,B:1,E:1,C:1}, formed=3 ✓

Shrinking phase:
Valid window found: "ADOBEC" (length=6)
left=1: DOBEC, formed=2 (lost A), stop shrinking

Continue expansion:
right=6: DOBECD, formed=2
...continue until next valid window...

Key Insights:
1. Use two hashmaps: one for target counts, one for window counts
2. Track "formed" - number of unique characters that have correct frequency
3. When formed == required, we have a valid window
4. Always try to shrink valid windows to find minimum

Time Complexity: O(|s| + |t|)
- Each character in s is visited at most twice (by left and right pointers)
- Building target count takes O(|t|)

Space Complexity: O(|s| + |t|)
- HashMap for target characters: O(|t|)
- HashMap for window characters: O(|s|) in worst case

Edge Cases:
1. s shorter than t: impossible, return ""
2. t not in s: no valid window, return ""
3. s equals t: return s
4. Multiple valid windows: return the shortest one
5. Multiple shortest windows: return the first one found

Optimization notes:
- Use arrays instead of HashMap for ASCII characters (faster)
- Early termination when no more characters of t remain in s
- Can be extended to handle Unicode characters
*/
```

### 19. Valid Anagram

```java
import java.util.*;

/**
 * Problem: Check if two strings are anagrams
 * 
 * Multiple approaches with different trade-offs
 */
public class ValidAnagram {
    
    // Approach 1: Character frequency counting - Most efficient
    // Time: O(n), Space: O(1) for fixed alphabet size
    public boolean isAnagram1(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        
        int[] charCount = new int[26]; // For lowercase English letters
        
        // Count characters in s and t
        for (int i = 0; i < s.length(); i++) {
            charCount[s.charAt(i) - 'a']++;
            charCount[t.charAt(i) - 'a']--;
        }
        
        // Check if all counts are zero
        for (int count : charCount) {
            if (count != 0) {
                return false;
            }
        }
        
        return true;
    }
    
    // Approach 2: Using HashMap - Works for Unicode
    // Time: O(n), Space: O(1) for fixed alphabet size
    public boolean isAnagram2(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        
        Map<Character, Integer> charCount = new HashMap<>();
        
        // Count characters in s
        for (char c : s.toCharArray()) {
            charCount.put(c, charCount.getOrDefault(c, 0) + 1);
        }
        
        // Subtract
 characters in t
        for (char c : t.toCharArray()) {
            if (!charCount.containsKey(c)) {
                return false;
            }
            charCount.put(c, charCount.get(c) - 1);
            if (charCount.get(c) == 0) {
                charCount.remove(c);
            }
        }
        
        return charCount.isEmpty();
    }
    
    // Approach 3: Sorting - Simple but less efficient
    // Time: O(n log n), Space: O(n) for sorting
    public boolean isAnagram3(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        
        char[] sArray = s.toCharArray();
        char[] tArray = t.toCharArray();
        
        Arrays.sort(sArray);
        Arrays.sort(tArray);
        
        return Arrays.equals(sArray, tArray);
    }
    
    // Approach 4: One-liner using streams (least efficient)
    public boolean isAnagram4(String s, String t) {
        return s.length() == t.length() && 
               s.chars().sorted().collect(StringBuilder::new, 
                   StringBuilder::appendCodePoint, StringBuilder::append).toString()
               .equals(t.chars().sorted().collect(StringBuilder::new, 
                   StringBuilder::appendCodePoint, StringBuilder::append).toString());
    }
}

/*
Algorithm Comparison:

Approach 1 (Character Array):
- Best for lowercase English letters
- O(1) space complexity (fixed size array)
- Fastest execution due to array access

Approach 2 (HashMap):
- Works for any character set (Unicode)
- O(k) space where k is number of unique characters
- Slightly slower due to HashMap operations

Approach 3 (Sorting):
- Simple to understand and implement
- O(n log n) time complexity
- O(n) space for sorting

Approach 4 (Streams):
- Very concise but poor performance
- Creates multiple intermediate objects
- Not recommended for production code

Example walkthrough (Approach 1):
s = "anagram", t = "nagaram"

Initialize: charCount = [0,0,0,...,0] (26 zeros)

Process s:
'a': charCount[0]++ → [1,0,0,...]
'n': charCount[13]++ → [1,0,0,...,1,...]
'a': charCount[0]++ → [2,0,0,...,1,...]
'g': charCount[6]++ → [2,0,0,0,0,0,1,...]
'r': charCount[17]++ → [2,0,0,0,0,0,1,...,1,...]
'a': charCount[0]++ → [3,0,0,0,0,0,1,...,1,...]
'm': charCount[12]++ → [3,0,0,0,0,0,1,0,0,0,0,0,1,1,...]

Process t:
'n': charCount[13]-- → [3,0,0,0,0,0,1,0,0,0,0,0,1,0,...]
'a': charCount[0]-- → [2,0,0,0,0,0,1,0,0,0,0,0,1,0,...]
'g': charCount[6]-- → [2,0,0,0,0,0,0,0,0,0,0,0,1,0,...]
'a': charCount[0]-- → [1,0,0,0,0,0,0,0,0,0,0,0,1,0,...]
'r': charCount[17]-- → [1,0,0,0,0,0,0,0,0,0,0,0,1,0,...,0,...]
'a': charCount[0]-- → [0,0,0,0,0,0,0,0,0,0,0,0,1,0,...,0,...]
'm': charCount[12]-- → [0,0,0,0,0,0,0,0,0,0,0,0,0,0,...,0,...]

All counts are 0, so strings are anagrams.

When to use each approach:
1. Use array approach for known character set (e.g., lowercase letters)
2. Use HashMap for Unicode or unknown character sets
3. Use sorting for simplicity when performance isn't critical
4. Avoid streams approach in performance-critical code

Edge Cases:
- Empty strings: both empty → true
- Different lengths: → false
- Same string: → true
- Single character: check if same
*/
```

### 20. Group Anagrams

```java
import java.util.*;

/**
 * Problem: Group strings that are anagrams of each other
 * 
 * Optimal Approach: Use sorted string as key for group
ing
 * Time: O(n * k log k) where n is number of strings, k is max string length
 * Space: O(n * k)
 */
public class GroupAnagrams {
    
    // Approach 1: Sorting as key - Most intuitive
    public List<List<String>> groupAnagrams1(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        
        Map<String, List<String>> anagramGroups = new HashMap<>();
        
        for (String str : strs) {
            // Sort characters to create a key
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            
            // Add string to corresponding group
            anagramGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(anagramGroups.values());
    }
    
    // Approach 2: Character frequency as key - Better for long strings
    // Time: O(n * k), Space: O(n * k)
    public List<List<String>> groupAnagrams2(String[] strs) {
        Map<String, List<String>> anagramGroups = new HashMap<>();
        
        for (String str : strs) {
            String key = getFrequencyKey(str);
            anagramGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(anagramGroups.values());
    }
    
    private String getFrequencyKey(String str) {
        int[] charCount = new int[26];
        
        // Count character frequencies
        for (char c : str.toCharArray()) {
            charCount[c - 'a']++;
        }
        
        // Build key from frequencies
        StringBuilder key = new StringBuilder();
        for (int i = 0; i < 26; i++) {
            if (charCount[i] > 0) {
                key.append((char)('a' + i)).append(charCount[i]);
            }
        }
        
        return key.toString();
    }
    
    // Approach 3: Prime number encoding - Most efficient for small strings
    // Time: O(n * k), Space: O(n * k)
    public List<List<String>> groupAnagrams3(String[] strs) {
        // Prime numbers for each letter
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
        
        Map<Long, List<String>> anagramGroups = new HashMap<>();
        
        for (String str : strs) {
            long key = 1;
            for (char c : str.toCharArray()) {
                key *= primes[c - 'a'];
            }
            anagramGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(anagramGroups.values());
    }
    
    // Approach 4: Using character count array as key
    public List<List<String>> groupAnagrams4(String[] strs) {
        Map<String, List<String>> anagramGroups = new HashMap<>();
        
        for (String str : strs) {
            int[] charCount = new int[26];
            for (char c : str.toCharArray()) {
                charCount[c - 'a']++;
            }
            
            // Convert array to string key
            String key = Arrays.toString(charCount);
            anagramGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(anagramGroups.values());
    }
}

/*
Algorithm Explanation:

The key insight is that anagrams will have the same "signature":
1. Same characters when sorted
2. Same character frequency distribution
3. Same prime factorization (unique for each character combination)

Example: ["eat","tea","tan","ate","nat","bat"]

Approach 1 (Sorting):
"eat" → "aet"
"tea" → "aet" 
"tan" → "ant"
"ate" → "aet"
"nat" → "ant"
"bat" → "abt"

Groups: {"aet": ["eat","tea","ate"], "ant": ["tan","nat"], "abt": ["bat"]}

Approach 2 (Frequency):
"eat" → "a1e1t1"
"tea" → "a1e1t1"
"tan" → "a1n1t1"
"ate" → "a1e1t1"
"nat" → "a1n1t1"
"bat" → "a1b1t1"

Approach 3 (Prime encoding):
"eat" → 2*29*83 = 4814
"tea" → 2*29*83 = 4814
"tan" → 2*31*83 = 5146
...

Approach Comparison:

1. Sorting (Approach 1):
   - Time
: O(n * k log k)
   - Space: O(n * k)
   - Most intuitive and commonly used
   - Works for any character set

2. Frequency counting (Approach 2):
   - Time: O(n * k)
   - Space: O(n * k)
   - Better for long strings
   - Only works for known character set

3. Prime encoding (Approach 3):
   - Time: O(n * k)
   - Space: O(n * k)
   - Most efficient for computation
   - Risk of overflow for very long strings
   - Limited to predefined character set

4. Array as key (Approach 4):
   - Time: O(n * k)
   - Space: O(n * k)
   - Simple implementation
   - Arrays.toString() creates readable keys

When to use each:
- General case: Use sorting (Approach 1)
- Long strings with small alphabet: Use frequency (Approach 2)
- Performance critical with short strings: Use prime encoding (Approach 3)
- Debugging/readable keys: Use array approach (Approach 4)

Edge Cases:
- Empty array: return empty list
- Single string: return list with one group
- All strings are anagrams: return single group
- No anagrams: return list where each group has one string
- Empty strings: group together

Time Complexity Analysis:
- n = number of strings
- k = maximum length of string
- Sorting: O(n * k log k) due to sorting each string
- Frequency/Prime: O(n * k) as we scan each character once

Space Complexity: O(n * k) for storing all strings in groups
*/
```

### 21. Valid Parentheses

```java
import java.util.*;

/**
 * Problem: Check if parentheses string is valid (properly matched and nested)
 * 
 * Optimal Approach: Stack for matching pairs
 * Time: O(n), Space: O(n)
 */
public class ValidParentheses {
    
    // Approach 1: Using Stack - Most intuitive
    public boolean isValid1(String s) {
        if (s == null || s.length() % 2 != 0) {
            return false;
        }
        
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            // Push opening brackets onto stack
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            }
            // Check closing brackets
            else if (c == ')' || c == ']' || c == '}') {
                if (stack.isEmpty()) {
                    return false; // No matching opening bracket
                }
                
                char top = stack.pop();
                if (!isMatchingPair(top, c)) {
                    return false; // Mismatched pair
                }
            }
        }
        
        return stack.isEmpty(); // All brackets should be matched
    }
    
    private boolean isMatchingPair(char open, char close) {
        return (open == '(' && close == ')') ||
               (open == '[' && close == ']') ||
               (open == '{' && close == '}');
    }
    
    // Approach 2: Using HashMap for cleaner mapping
    public boolean isValid2(String s) {
        if (s.length() % 2 != 0) return false;
        
        Map<Character, Character> mapping = new HashMap<>();
        mapping.put(')', '(');
        mapping.put(']', '[');
        mapping.put('}', '{');
        
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (mapping.containsKey(c)) {
                // Closing bracket
                if (stack.isEmpty() || stack.pop() != mapping.get(c)) {
                    return false;
                }
            } else {
                // Opening bracket
                stack.push(c);
            }
        }
        
        return stack.isEmpty();
    }
    
    // Approach 3: Using array as stack (more efficient)
    public boolean isValid3(String s) {
        if (s.length() % 2 != 0) return false;
        
        char[] stack = new char[s.length()];
        int top = -1;
        
        for (char c : s.toCharArray()) {
            switch (c) {
                case '(':
                case '[':
                case '{':
                    stack[++top] = c;
                    break;
                case ')':
                    if (top == -1 || stack[top--] != '(') return false;
                    break;
                case ']':
                    if (top == -1 || stack[top--] != '[') return false;
                    break;
                case '}':
                    if (top == -1 || stack[top--] != '{') return false;
                    break;
                default:
                    // Invalid character
                    return false;
            }
        }
        
        return top == -1;
    }
    
    // Approach 4: String replacement (
inefficient but interesting)
    public boolean isValid4(String s) {
        while (s.contains("()") || s.contains("[]") || s.contains("{}")) {
            s = s.replace("()", "").replace("[]", "").replace("{}", "");
        }
        return s.isEmpty();
    }
}

/*
Algorithm Explanation (Stack approach):

1. Use stack to keep track of opening brackets
2. When encountering opening bracket: push to stack
3. When encountering closing bracket: 
   - Check if stack is empty (no matching opening)
   - Pop from stack and verify it matches current closing bracket
4. At the end, stack should be empty (all brackets matched)

Example: s = "([{}])"

Step by step:
c='(': opening bracket, push to stack: ['(']
c='[': opening bracket, push to stack: ['(', '[']  
c='{': opening bracket, push to stack: ['(', '[', '{']
c='}': closing bracket, pop '{', matches ✓, stack: ['(', '[']
c=']': closing bracket, pop '[', matches ✓, stack: ['(']
c=')': closing bracket, pop '(', matches ✓, stack: []

Stack is empty → valid

Example: s = "([)]"
c='(': push, stack: ['(']
c='[': push, stack: ['(', '[']
c=')': pop '[', doesn't match ')' → invalid

Key Insights:
1. Stack naturally handles the "last opened, first closed" property
2. Every closing bracket must have a corresponding opening bracket
3. The most recent unmatched opening bracket should match current closing bracket
4. Final stack must be empty for valid string

Approach Comparison:

1. Stack with helper method:
   - Most readable and maintainable
   - Easy to extend for more bracket types
   - Standard Stack operations

2. HashMap mapping:
   - Cleaner code with data-driven approach
   - Easy to modify bracket types
   - Slightly more memory for HashMap

3. Array as stack:
   - Most efficient (no object creation)
   - Faster than Stack class
   - More complex code

4. String replacement:
   - Interesting but very inefficient O(n²)
   - Creates many intermediate strings
   - Not recommended for production

Time Complexity: O(n) - scan string once
Space Complexity: O(n) - worst case all opening brackets

Edge Cases:
- Empty string: valid (vacuously true)
- Odd length: invalid (can't be balanced)
- Only opening brackets: invalid
- Only closing brackets: invalid
- Mixed valid/invalid: check each character
- Single bracket: invalid

Common mistakes:
- Forgetting to check if stack is empty before popping
- Not checking if stack is empty at the end
- Not handling mismatched bracket types
- Not considering odd-length strings
*/
```

### 22. Valid Palindrome

```java
/**
 * Problem: Check if string is palindrome (ignoring non-alphanumeric and case)
 * 
 * Optimal Approach: Two pointers from both ends
 * Time: O(n), Space: O(1)
 */
public class ValidPalindrome {
    
    // Approach 1: Two pointers - Most efficient
    public boolean isPalindrome1(String s) {
        if (s == null) return false;
        
        int left = 0;
        int right = s.length() - 1;
        
        while (left < right) {
            // Skip non-alphanumeric characters from left
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            
            // Skip non-alphanumeric characters from right
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            
            // Compare characters (case-insensitive)
            if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            
            left++;
            right--;
        }
        

        return true;
    }
    
    // Approach 2: Clean string first, then check
    public boolean isPalindrome2(String s) {
        if (s == null) return false;
        
        // Clean the string: keep only alphanumeric, convert to lowercase
        StringBuilder cleaned = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                cleaned.append(Character.toLowerCase(c));
            }
        }
        
        String cleanStr = cleaned.toString();
        String reversed = cleaned.reverse().toString();
        
        return cleanStr.equals(reversed);
    }
    
    // Approach 3: Using StringBuilder reverse
    public boolean isPalindrome3(String s) {
        StringBuilder alphanumeric = new StringBuilder();
        
        for (char c : s.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                alphanumeric.append(Character.toLowerCase(c));
            }
        }
        
        return alphanumeric.toString().equals(alphanumeric.reverse().toString());
    }
    
    // Approach 4: Recursive solution
    public boolean isPalindrome4(String s) {
        return isPalindromeHelper(s, 0, s.length() - 1);
    }
    
    private boolean isPalindromeHelper(String s, int left, int right) {
        // Base case
        if (left >= right) return true;
        
        // Skip non-alphanumeric from left
        if (!Character.isLetterOrDigit(s.charAt(left))) {
            return isPalindromeHelper(s, left + 1, right);
        }
        
        // Skip non-alphanumeric from right
        if (!Character.isLetterOrDigit(s.charAt(right))) {
            return isPalindromeHelper(s, left, right - 1);
        }
        
        // Compare current characters
        if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
            return false;
        }
        
        // Check remaining substring
        return isPalindromeHelper(s, left + 1, right - 1);
    }
    
    // Approach 5: Using regex (least efficient)
    public boolean isPalindrome5(String s) {
        String cleaned = s.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        return cleaned.equals(new StringBuilder(cleaned).reverse().toString());
    }
}

/*
Algorithm Explanation (Two Pointers):

1. Use two pointers starting from both ends
2. Skip non-alphanumeric characters
3. Compare characters in case-insensitive manner
4. Move pointers inward until they meet

Example: s = "A man, a plan, a canal: Panama"

Cleaned version: "amanaplanacanalpanama"

Two pointer process:
left=0 ('A'), right=30 ('a'): 'a' == 'a' ✓, move inward
left=1 (' '), skip non-alphanumeric, left=2
left=2 ('m'), right=29 ('m'): 'm' == 'm' ✓, move inward
left=3 ('a'), right=28 ('a'): 'a' == 'a' ✓, move inward
...continue until left >= right

All comparisons match → palindrome

Approach Comparison:

1. Two Pointers (Approach 1):
   - Time: O(n), Space: O(1)
   - Most efficient, no extra string creation
   - Processes characters on-the-fly

2. Clean then Compare (Approach 2):
   - Time: O(n), Space: O(n)
   - Easier to understand
   - Creates intermediate string

3. StringBuilder (Approach 3):
   - Time: O(n), Space: O(n)
   - Concise but less efficient
   - Multiple string operations

4. Recursive (Approach 4):
   - Time: O(n), Space: O(n) for call stack
   - Elegant but uses more memory
   - Good for educational purposes

5. Regex (Approach 5):
   - Time: O(n), Space: O(n)
   - Very concise but slowest
   - Regex compilation overhead

Key Insights:
1. Palindrome means reads same forwards and backwards
2. Ignore case and non-alphanumeric characters
3. Two pointers avoid creating extra strings
4. Character.isLetterOrDigit() handles both letters and digits
5. Character.toLowerCase() handles case conversion

Edge Cases:
- Empty string: true (vacuously)
- Single character: true
- Only non-alphanumeric: true (empty after cleaning)
- All same character: true
- Mixed case: handle with toLowerCase
()

Optimization notes:
- Two pointers approach is optimal for space
- Avoid creating intermediate strings when possible
- Character methods are efficient for validation
- Early termination on first mismatch

Common mistakes:
- Forgetting to handle case sensitivity
- Not skipping non-alphanumeric characters
- Creating unnecessary intermediate strings
- Not handling empty or null strings
*/
```

### 23. Longest Palindromic Substring

```java
/**
 * Problem: Find the longest palindromic substring
 * 
 * Multiple approaches from brute force to optimal
 */
public class LongestPalindromicSubstring {
    
    // Approach 1: Expand Around Centers - Most intuitive and efficient
    // Time: O(n²), Space: O(1)
    public String longestPalindrome1(String s) {
        if (s == null || s.length() < 2) return s;
        
        int start = 0;
        int maxLength = 1;
        
        for (int i = 0; i < s.length(); i++) {
            // Check for odd-length palindromes (center at i)
            int len1 = expandAroundCenter(s, i, i);
            
            // Check for even-length palindromes (center between i and i+1)
            int len2 = expandAroundCenter(s, i, i + 1);
            
            int currentMax = Math.max(len1, len2);
            
            if (currentMax > maxLength) {
                maxLength = currentMax;
                start = i - (currentMax - 1) / 2;
            }
        }
        
        return s.substring(start, start + maxLength);
    }
    
    private int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1; // Length of palindrome
    }
    
    // Approach 2: Dynamic Programming
    // Time: O(n²), Space: O(n²)
    public String longestPalindrome2(String s) {
        if (s == null || s.length() < 2) return s;
        
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int start = 0;
        int maxLength = 1;
        
        // Every single character is a palindrome
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        
        // Check for palindromes of length 2
        for (int i = 0; i < n - 1; i++) {
            if (s.charAt(i) == s.charAt(i + 1)) {
                dp[i][i + 1] = true;
                start = i;
                maxLength = 2;
            }
        }
        
        // Check for palindromes of length 3 and more
        for (int length = 3; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                
                if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    start = i;
                    maxLength = length;
                }
            }
        }
        
        return s.substring(start, start + maxLength);
    }
    
    // Approach 3: Manacher's Algorithm - Most efficient
    // Time: O(n), Space: O(n)
    public String longestPalindrome3(String s) {
        if (s == null || s.length() < 2) return s;
        
        // Preprocess string: "abc" -> "^#a#b#c#$"
        String processed = preprocess(s);
        int n = processed.length();
        int[] P = new int[n]; // P[i] = radius of palindrome centered at i
        int center = 0, right = 0; // Current palindrome center and right boundary
        
        for (int i = 1; i < n - 1; i++) {
            int mirror = 2 * center - i; // Mirror of i with respect to center
            
            if (i < right) {
                P[i] = Math.min(right - i, P[mirror]);
            }
            
            // Try to expand palindrome centered at i
            try {
                while (processed.charAt(i + P[i] + 1) == processed.charAt(i - P[i] - 1)) {
                    P[i]++;
                }
            } catch (StringIndexOutOfBoundsException e) {
                // Reached boundary
            }
            
            // If palindrome centered at i extends past right, adjust center and right
            if (i + P[i] > right) {
                center = i;
                right = i + P[i];
            }
        }
        
        // Find the longest palindrome
        int maxLength = 0;
        int centerIndex = 0;
        for (int i = 1; i < n - 1;
 i++) {
            if (P[i] > maxLength) {
                maxLength = P[i];
                centerIndex = i;
            }
        }
        
        int start = (centerIndex - maxLength) / 2;
        return s.substring(start, start + maxLength);
    }
    
    private String preprocess(String s) {
        StringBuilder sb = new StringBuilder();
        sb.append("^");
        for (char c : s.toCharArray()) {
            sb.append("#").append(c);
        }
        sb.append("#$");
        return sb.toString();
    }
    
    // Approach 4: Brute Force (for comparison)
    // Time: O(n³), Space: O(1)
    public String longestPalindrome4(String s) {
        if (s == null || s.length() < 2) return s;
        
        String longest = "";
        
        for (int i = 0; i < s.length(); i++) {
            for (int j = i; j < s.length(); j++) {
                String substring = s.substring(i, j + 1);
                if (isPalindrome(substring) && substring.length() > longest.length()) {
                    longest = substring;
                }
            }
        }
        
        return longest;
    }
    
    private boolean isPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}

/*
Algorithm Explanation:

Approach 1 (Expand Around Centers):
For each possible center, expand outward while characters match.
Handle both odd-length (center at character) and even-length (center between characters) palindromes.

Example: s = "babad"
Center at index 1 ('a'): expand to get "bab" (length 3)
Center at index 2 ('b'): expand to get "aba" (length 3)
Both are valid, return first found: "bab"

Approach 2 (Dynamic Programming):
dp[i][j] = true if substring from i to j is palindrome
Build up from smaller substrings to larger ones.

Recurrence:
dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]

Approach 3 (Manacher's Algorithm):
Linear time algorithm using previously computed information.
Preprocess string to handle even/odd length uniformly.
Use symmetry property to avoid redundant comparisons.

Approach 4 (Brute Force):
Check every possible substring for palindrome property.
Inefficient but simple to understand.

Approach Comparison:

1. Expand Around Centers:
   - Time: O(n²), Space: O(1)
   - Easy to understand and implement
   - Good balance of efficiency and simplicity

2. Dynamic Programming:
   - Time: O(n²), Space: O(n²)
   - Clear recurrence relation
   - Uses more memory but systematic

3. Manacher's Algorithm:
   - Time: O(n), Space: O(n)
   - Most efficient but complex to implement
   - Best for very large inputs

4. Brute Force:
   - Time: O(n³), Space: O(1)
   - Simple but inefficient
   - Only for small inputs or educational purposes

Key Insights:
1. Palindromes have symmetry property
2. Can be odd-length (center at character) or even-length (center between characters)
3. Expand around center avoids redundant checks
4. DP builds solution bottom-up
5. Manacher's uses symmetry to achieve linear time

Edge Cases:
- Empty string: return ""
- Single character: return the character
- No palindrome longer than 1: return any single character
- Entire string is palindrome: return entire string
- Multiple palindromes of same max length: return first found

When to use each approach:
- General case: Expand around centers (good balance)
- Educational/interview: DP (shows algorithmic thinking)
- Performance critical: Manacher's (if worth the complexity)
- Quick prototype: Brute force (for small inputs)
*/
```

### 24. Palindromic Substrings

```java
/**
 * Problem: Count all palindromic substrings in a string
 * 
 * Optimal Approach: Expand around centers
 * Time: O(n²), Space: O(1)

 */
public class PalindromicSubstrings {
    
    // Approach 1: Expand Around Centers - Most efficient
    public int countSubstrings1(String s) {
        if (s == null || s.length() == 0) return 0;
        
        int count = 0;
        
        for (int i = 0; i < s.length(); i++) {
            // Count odd-length palindromes centered at i
            count += expandAroundCenter(s, i, i);
            
            // Count even-length palindromes centered between i and i+1
            count += expandAroundCenter(s, i, i + 1);
        }
        
        return count;
    }
    
    private int expandAroundCenter(String s, int left, int right) {
        int count = 0;
        
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
        
        return count;
    }
    
    // Approach 2: Dynamic Programming
    // Time: O(n²), Space: O(n²)
    public int countSubstrings2(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int count = 0;
        
        // Every single character is a palindrome
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
            count++;
        }
        
        // Check for palindromes of length 2
        for (int i = 0; i < n - 1; i++) {
            if (s.charAt(i) == s.charAt(i + 1)) {
                dp[i][i + 1] = true;
                count++;
            }
        }
        
        // Check for palindromes of length 3 and more
        for (int length = 3; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                
                if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    count++;
                }
            }
        }
        
        return count;
    }
    
    // Approach 3: Manacher's Algorithm adaptation
    // Time: O(n), Space: O(n)
    public int countSubstrings3(String s) {
        String processed = preprocess(s);
        int n = processed.length();
        int[] P = new int[n];
        int center = 0, right = 0;
        int totalCount = 0;
        
        for (int i = 1; i < n - 1; i++) {
            int mirror = 2 * center - i;
            
            if (i < right) {
                P[i] = Math.min(right - i, P[mirror]);
            }
            
            // Try to expand palindrome centered at i
            while (i + P[i] + 1 < n && i - P[i] - 1 >= 0 && 
                   processed.charAt(i + P[i] + 1) == processed.charAt(i - P[i] - 1)) {
                P[i]++;
            }
            
            // If palindrome centered at i extends past right, adjust center and right
            if (i + P[i] > right) {
                center = i;
                right = i + P[i];
            }
            
            // Count palindromes: P[i] gives us the radius
            // For processed string, each unit radius corresponds to one palindrome in original
            totalCount += (P[i] + 1) / 2;
        }
        
        return totalCount;
    }
    
    private String preprocess(String s) {
        StringBuilder sb = new StringBuilder();
        sb.append("^");
        for (char c : s.toCharArray()) {
            sb.append("#").append(c);
        }
        sb.append("#$");
        return sb.toString();
    }
    
    // Approach 4: Brute Force (for comparison)
    // Time: O(n³), Space: O(1)
    public int countSubstrings4(String s) {
        int count = 0;
        
        for (int i = 0; i < s.length(); i++) {
            for (int j = i; j < s.length(); j++) {
                if (isPalindrome(s, i, j)) {
                    count++;
                }
            }
        }
        
        return count;
    }
    
    private boolean isPalindrome(String s, int start, int end) {
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}

/*
Algorithm Explanation:

Approach 1 (Expand Around Centers):
For each possible center, expand outward and count all palindromes found.
Each expansion that maintains palindrome property contributes 1 to count.

Example: s = "abc"
Center at 0 ('a'): "a" → count = 1
Center between 0,1: no palindrome → count = 0  
Center at 1 ('b'): "b" → count = 1
Center between 1,2: no palindrome → count = 0
Center at 2 ('c'): "c" → count = 1
Total: 3

Example: s = "aaa"
Center at 0: "a" → count = 1
Center between 0,1: "aa" → count = 1
Center at 1: "a", "aaa" → count = 2  
Center between 1,2: "aa" → count = 1
Center at 2: "a" → count = 1
Total: 6 palindromes: "a"(3), "aa"(2), "aaa"(1)

Approach 2 (Dynamic Programming):
Build table where dp[i][j] indicates if substring from i to j is palindrome.
Count all
 true entries in the table.

Recurrence relation:
- dp[i][i] = true (single characters)
- dp[i][i+1] = (s[i] == s[i+1]) (two characters)
- dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1] (longer substrings)

Approach 3 (Manacher's Algorithm):
Linear time algorithm that counts palindromes while finding them.
More complex but most efficient for large inputs.

Approach 4 (Brute Force):
Check every possible substring for palindrome property.
Simple but inefficient.

Approach Comparison:

1. Expand Around Centers:
   - Time: O(n²), Space: O(1)
   - Most practical and intuitive
   - Good performance with minimal space

2. Dynamic Programming:
   - Time: O(n²), Space: O(n²)
   - Systematic approach
   - Uses more memory but clear logic

3. Manacher's Algorithm:
   - Time: O(n), Space: O(n)
   - Most efficient for very large strings
   - Complex implementation

4. Brute Force:
   - Time: O(n³), Space: O(1)
   - Simple but slow
   - Only for educational purposes

Key Insights:
1. Every single character is a palindrome
2. Palindromes can be odd-length (centered at character) or even-length (centered between characters)
3. Expanding around centers avoids redundant work
4. Each valid expansion contributes exactly one palindrome to the count

Mathematical insight:
For string of length n:
- Minimum palindromes: n (all single characters)
- Maximum palindromes: n(n+1)/2 (when all characters are same)

Edge Cases:
- Empty string: 0 palindromes
- Single character: 1 palindrome
- All same characters: n(n+1)/2 palindromes
- All different characters: n palindromes (only single chars)

Optimization notes:
- Expand around centers is optimal balance of simplicity and efficiency
- Early termination when characters don't match
- No need to store palindromes, just count them
- Can be extended to return actual palindromic substrings if needed
*/
```

That completes all 9 String problems! Each solution includes:

1. **Multiple approaches** from brute force to optimal
2. **Detailed algorithm explanations** with step-by-step examples
3. **Time and space complexity analysis** for each approach
4. **Key insights** and problem-solving strategies
5. **Edge case handling** and when to use each approach
6. **Code comments** explaining the logic

The string problems covered important patterns like:
- **Sliding Window** (Longest Substring, Character Replacement, Minimum Window)
- **Two Pointers** (Valid Palindrome)
- **HashMap/Frequency Counting** (Anagrams, Group Anagrams)
- **Stack** (Valid Parentheses)
- **Expand Around Centers** (Palindromic problems)
- **Dynamic Programming** (Palindromic Substring)

--------------------------------------------------------------------------



## Tree Problems Solutions

### 25. Maximum Depth of Binary Tree

```java
import java.util.*;

/**
 * Problem: Find the maximum depth (height) of a binary tree
 * 
 * Multiple approaches: Recursive, Iterative (DFS & BFS)
 */
public class MaximumDepthOfBinaryTree {
    
    // Definition for a binary tree node
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive DFS - Most intuitive
    // Time: O(n), Space: O(h) where h is height (O(n) worst case for skewed tree)
    public int maxDepth1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        int leftDepth = maxDepth1(root.left);
        int rightDepth = maxDepth1(root.right);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
    
    // Approach 2: Iterative DFS using Stack
    // Time: O(n), Space: O(h)
    public int maxDepth2(TreeNode root) {
        if (root == null) return 0;
        
        Stack<TreeNode> nodeStack = new Stack<>();
        Stack<Integer> depthStack = new Stack<>();
        
        nodeStack.push(root);
        depthStack.push(1);
        
        int maxDepth = 0;
        
        while (!nodeStack.isEmpty()) {
            TreeNode node = nodeStack.pop();
            int currentDepth = depthStack.pop();
            
            maxDepth = Math.max(maxDepth, currentDepth);
            
            if (node.left != null) {
                nodeStack.push(node.left);
                depthStack.push(currentDepth + 1);
            }
            
            if (node.right != null) {
                nodeStack.push(node.right);
                depthStack.push(currentDepth + 1);
            }
        }
        
        return maxDepth;
    }
    
    // Approach 3: BFS (Level Order Traversal)
    // Time: O(n), Space: O(w) where w is maximum width
    public int maxDepth3(TreeNode root) {
        if (root == null) return 0;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            depth++;
            
            // Process all nodes at current level
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        
        return depth;
    }
    
    // Approach 4: Using Pair class for cleaner iterative solution
    public int maxDepth4(TreeNode root) {
        if (root == null) return 0;
        
        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 1));
        
        int maxDepth = 0;
        
        while (!stack.isEmpty()) {
            Pair<TreeNode, Integer> current = stack.pop();
            TreeNode node = current.getKey();
            int depth = current.getValue();
            
            maxDepth = Math.max(maxDepth, depth);
            
            if (node.left != null) {
                stack.push(new Pair<>(node.left, depth + 1));
            }
            if (node.right != null) {
                stack.push(new Pair<>(node.right, depth + 1));
            }
        }
        
        return maxDepth;
    }
    
    // Helper class for Approach 4
    class Pair<K, V> {
        private K key;
        private V value;
        
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }
        
        public K getKey() { return key; }
        public V getValue() { return value; }
    }
}

/*
Algorithm Explanation:

Approach 1 (Recursive DFS):
Base case: null node has depth 0
Recursive case: depth = 1 + max(left_depth, right_depth)

Example tree:
    3
   / \
  9   20
     /  \
    15   7

Recursion trace:
maxDepth(3): 
  maxDepth(9): 
    maxDepth(null)
 = 0, maxDepth(null) = 0
    return max(0,0) + 1 = 1
  maxDepth(20):
    maxDepth(15): return 1
    maxDepth(7): return 1
    return max(1,1) + 1 = 2
  return max(1,2) + 1 = 3

Approach 2 (Iterative DFS):
Use stack to simulate recursion
Store both node and its depth
Track maximum depth seen

Approach 3 (BFS):
Process tree level by level
Count number of levels
Each level increases depth by 1

Approach 4 (Pair-based):
Cleaner version of iterative DFS
Uses Pair to bundle node with depth

Approach Comparison:

1. Recursive DFS:
   - Most intuitive and concise
   - Natural tree traversal
   - Call stack overhead

2. Iterative DFS:
   - Avoids recursion overhead
   - More control over traversal
   - Slightly more complex

3. BFS (Level Order):
   - Natural for depth calculation
   - Good for wide trees
   - Uses queue instead of stack

4. Pair-based:
   - Cleaner code structure
   - Object creation overhead
   - More readable

Time Complexity: O(n) for all approaches - must visit every node
Space Complexity:
- Recursive: O(h) where h is height
- Iterative DFS: O(h) for stack
- BFS: O(w) where w is maximum width
- Worst case: O(n) for completely unbalanced tree

Key Insights:
1. Tree depth = longest path from root to leaf
2. Recursive solution naturally follows tree structure
3. Iterative solutions avoid potential stack overflow
4. BFS processes level by level, making depth calculation natural

Edge Cases:
- Empty tree (null root): depth = 0
- Single node: depth = 1
- Linear tree (linked list): depth = n
- Complete binary tree: depth = log(n) + 1

When to use each approach:
- Recursive: Most interviews, clean and intuitive
- Iterative DFS: When recursion depth might be too large
- BFS: When you need level-by-level processing
- Pair-based: When code readability is priority

Common mistakes:
- Forgetting base case for recursion
- Not handling null nodes properly
- Off-by-one errors in depth calculation
- Stack overflow for very deep trees with recursive approach
*/
```

### 26. Same Tree

```java
/**
 * Problem: Check if two binary trees are identical
 * 
 * Multiple approaches: Recursive, Iterative
 */
public class SameTree {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive - Most elegant
    // Time: O(min(m,n)), Space: O(min(m,n))
    public boolean isSameTree1(TreeNode p, TreeNode q) {
        // Base cases
        if (p == null && q == null) {
            return true; // Both null
        }
        
        if (p == null || q == null) {
            return false; // One null, one not null
        }
        
        // Check current nodes and recursively check subtrees
        return (p.val == q.val) && 
               isSameTree1(p.left, q.left) && 
               isSameTree1(p.right, q.right);
    }
    
    // Approach 2: Iterative using Stack
    // Time: O(min(m,n)), Space: O(min(m,n))
    public boolean isSameTree2(TreeNode p, TreeNode q) {
        Stack<TreeNode> stack = new Stack<>();
        stack.push(p);
        stack.push(q);
        
        while (!stack.isEmpty()) {
            TreeNode node1 = stack.pop();
            TreeNode node2 = stack.pop();
            
            // Both null - continue
            if (node1 == null && node2 == null) {
                continue;
            }
            
            // One null, one not null - not same
            if (node1 == null || node2 == null) {
                return false;
            }
            
            // Different values - not same
            if (node1.val != node2.val) {
                return false;
            }
            
            // Add children to stack for comparison

            stack.push(node1.left);
            stack.push(node2.left);
            stack.push(node1.right);
            stack.push(node2.right);
        }
        
        return true;
    }
    
    // Approach 3: Using Queue (BFS style)
    // Time: O(min(m,n)), Space: O(min(m,n))
    public boolean isSameTree3(TreeNode p, TreeNode q) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(p);
        queue.offer(q);
        
        while (!queue.isEmpty()) {
            TreeNode node1 = queue.poll();
            TreeNode node2 = queue.poll();
            
            if (node1 == null && node2 == null) {
                continue;
            }
            
            if (node1 == null || node2 == null || node1.val != node2.val) {
                return false;
            }
            
            queue.offer(node1.left);
            queue.offer(node2.left);
            queue.offer(node1.right);
            queue.offer(node2.right);
        }
        
        return true;
    }
    
    // Approach 4: Serialize and Compare
    // Time: O(m+n), Space: O(m+n)
    public boolean isSameTree4(TreeNode p, TreeNode q) {
        return serialize(p).equals(serialize(q));
    }
    
    private String serialize(TreeNode root) {
        if (root == null) {
            return "null";
        }
        
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }
    
    // Approach 5: Preorder traversal comparison
    // Time: O(m+n), Space: O(m+n)
    public boolean isSameTree5(TreeNode p, TreeNode q) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        
        preorderTraversal(p, list1);
        preorderTraversal(q, list2);
        
        return list1.equals(list2);
    }
    
    private void preorderTraversal(TreeNode root, List<Integer> list) {
        if (root == null) {
            list.add(null); // Important: add null to distinguish structure
            return;
        }
        
        list.add(root.val);
        preorderTraversal(root.left, list);
        preorderTraversal(root.right, list);
    }
}

/*
Algorithm Explanation:

Two trees are identical if:
1. Both are null, OR
2. Both have same value at root AND
3. Left subtrees are identical AND
4. Right subtrees are identical

Approach 1 (Recursive):
Most natural approach following tree structure.
Base cases handle null scenarios.
Recursive calls check subtrees.

Example:
Tree p:    Tree q:
   1         1
  / \       / \
 2   3     2   3

Comparison:
isSameTree(1,1): values equal, check subtrees
  isSameTree(2,2): values equal, check subtrees
    isSameTree(null,null): both null, return true
    isSameTree(null,null): both null, return true
    return true
  isSameTree(3,3): values equal, check subtrees
    isSameTree(null,null): both null, return true
    isSameTree(null,null): both null, return true
    return true
  return true

Approach 2 (Iterative Stack):
Simulates recursion using explicit stack.
Push corresponding nodes together.
Pop and compare pairs.

Approach 3 (Iterative Queue):
BFS-style comparison.
Process nodes level by level.
Same logic as stack but different order.

Approach 4 (Serialization):
Convert both trees to strings.
Compare string representations.
Must handle null nodes properly.

Approach 5 (Traversal Lists):
Generate traversal sequences.
Compare sequences for equality.
Must include null markers for structure.

Approach Comparison:

1. Recursive:
   - Most intuitive and clean
   - Natural tree processing
   - Potential stack overflow for deep trees

2. Iterative Stack:
   - Avoids recursion overhead
   - More control over traversal
   - Slightly more complex

3. Iterative Queue:
   - BFS-style processing
   - Good for wide trees
   - Similar complexity to stack

4. Serialization:
   - Simple concept
   - Higher space usage
   - String operations overhead

5. Traversal Lists:
   - Easy to understand
   - Requires full traversal of both trees
   - Higher space complexity

Time Complexity: O(min(m,n)) for approaches 1-3
- We
 stop as soon as we find a difference
- In worst case, visit all nodes of smaller tree

Space Complexity:
- Recursive: O(min(m,n)) for call stack
- Iterative: O(min(m,n)) for stack/queue
- Serialization: O(m+n) for strings
- Traversal: O(m+n) for lists

Key Insights:
1. Early termination when difference found
2. Must handle null nodes carefully
3. Structure matters as much as values
4. Recursive solution mirrors problem definition

Edge Cases:
- Both trees null: true
- One tree null, other not: false
- Same structure, different values: false
- Different structure, same values: false
- Single node trees: compare values
- Identical trees: true

Optimization notes:
- Recursive approach is most efficient for balanced trees
- Iterative approaches avoid potential stack overflow
- Early termination saves unnecessary comparisons
- Serialization approach is less efficient but conceptually simple

Common mistakes:
- Not handling null nodes properly
- Forgetting to check both structure and values
- Not using early termination
- Incorrect null comparisons (using == vs proper null checks)
*/
```

### 27. Invert Binary Tree

```java
import java.util.*;

/**
 * Problem: Invert/flip a binary tree (mirror image)
 * 
 * Multiple approaches: Recursive, Iterative
 */
public class InvertBinaryTree {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive - Most elegant
    // Time: O(n), Space: O(h) where h is height
    public TreeNode invertTree1(TreeNode root) {
        if (root == null) {
            return null;
        }
        
        // Swap left and right children
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        
        // Recursively invert subtrees
        invertTree1(root.left);
        invertTree1(root.right);
        
        return root;
    }
    
    // Approach 2: Recursive with post-order style
    // Time: O(n), Space: O(h)
    public TreeNode invertTree2(TreeNode root) {
        if (root == null) {
            return null;
        }
        
        // First invert subtrees, then swap
        TreeNode left = invertTree2(root.left);
        TreeNode right = invertTree2(root.right);
        
        root.left = right;
        root.right = left;
        
        return root;
    }
    
    // Approach 3: Iterative using Stack (DFS)
    // Time: O(n), Space: O(h)
    public TreeNode invertTree3(TreeNode root) {
        if (root == null) return null;
        
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            
            // Swap children
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;
            
            // Add children to stack for processing
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
        
        return root;
    }
    
    // Approach 4: Iterative using Queue (BFS)
    // Time: O(n), Space: O(w) where w is maximum width
    public TreeNode invertTree4(TreeNode root) {
        if (root == null) return null;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            
            // Swap children
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;
            
            // Add children to queue for processing
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        
        return root;
    }
    
    // Approach 5: Morris Traversal (constant space)
    // Time: O(n), Space: O(1)
    public TreeNode invertTree5(TreeNode root) {
        TreeNode current = root;
        
        while (current != null) {
            if (current.left == null) {
                // Swap and move to right
                TreeNode temp = current.left;
                current.left = current.right;
                current.right = temp;
                current = current.left; // This is now the original right
            } else {
                // Find inorder predecessor
                TreeNode predecessor = current.left;
                while (predecessor.right != null && predecessor.right != current) {
                    predecessor
 = predecessor.right;
                }
                
                if (predecessor.right == null) {
                    // Make current as right child of predecessor
                    predecessor.right = current;
                    current = current.left;
                } else {
                    // Revert the changes
                    predecessor.right = null;
                    
                    // Swap children
                    TreeNode temp = current.left;
                    current.left = current.right;
                    current.right = temp;
                    
                    current = current.left; // Move to original right
                }
            }
        }
        
        return root;
    }
}

/*
Algorithm Explanation:

Inverting a binary tree means swapping left and right children of every node.

Original:     Inverted:
    4             4
   / \           / \
  2   7         7   2
 / \ / \       / \ / \
1  3 6  9     9  6 3  1

Approach 1 (Recursive - Pre-order):
1. Swap children of current node
2. Recursively invert left subtree
3. Recursively invert right subtree

Example trace for tree [4,2,7,1,3,6,9]:
invertTree(4):
  swap: left=7, right=2
  invertTree(7): // originally right child
    swap: left=9, right=6
    invertTree(9): return 9
    invertTree(6): return 6
    return 7
  invertTree(2): // originally left child
    swap: left=3, right=1
    invertTree(3): return 3
    invertTree(1): return 1
    return 2
  return 4

Approach 2 (Recursive - Post-order):
1. First recursively invert subtrees
2. Then swap the inverted subtrees
3. More functional programming style

Approach 3 (Iterative Stack - DFS):
1. Use stack to traverse tree
2. For each node, swap children
3. Add children to stack for processing
4. Simulates recursive approach

Approach 4 (Iterative Queue - BFS):
1. Use queue for level-order traversal
2. Process nodes level by level
3. Swap children of each node
4. Add children to queue

Approach 5 (Morris Traversal):
1. Constant space complexity
2. Uses threading technique
3. More complex but space-efficient
4. Modifies tree structure temporarily

Approach Comparison:

1. Recursive Pre-order:
   - Most intuitive and clean
   - Natural tree processing
   - Easy to understand and implement

2. Recursive Post-order:
   - Functional style
   - Cleaner separation of concerns
   - Same complexity as pre-order

3. Iterative Stack (DFS):
   - Avoids recursion overhead
   - Good for very deep trees
   - More control over traversal

4. Iterative Queue (BFS):
   - Level-by-level processing
   - Good for wide trees
   - Natural breadth-first approach

5. Morris Traversal:
   - Constant space complexity
   - Most complex implementation
   - Rarely needed in practice

Time Complexity: O(n) for all approaches - must visit every node
Space Complexity:
- Recursive: O(h) where h is height (call stack)
- Iterative: O(h) for stack, O(w) for queue
- Morris: O(1) constant space

Key Insights:
1. Inversion is swapping left and right children
2. Must apply recursively to all nodes
3. Order of operations doesn't matter much
4. Tree structure is preserved, just mirrored

Edge Cases:
- Empty tree (null): return null
- Single node: return same node
- Linear tree: becomes reverse linear tree
- Complete binary tree: remains complete

Visual Example:
Before:        After:
   1             1
  / \           / \
 2   3         3   2
    /           \
   4             4

Steps:
1. Swap children of node 1: left=3, right=2
2. Process node 3: no children to swap
3. Process node 2: swap null and 4
4. Process node 4: no
 children to swap

When to use each approach:
- Recursive: Most interviews and general use
- Iterative Stack: When recursion depth might be too large
- Iterative Queue: When you prefer BFS or need level processing
- Morris: When space is extremely constrained (rare)

Common mistakes:
- Forgetting to handle null nodes
- Not swapping children properly
- Infinite recursion due to incorrect base case
- Modifying tree structure incorrectly in Morris approach
*/
```

### 28. Binary Tree Maximum Path Sum

```java
/**
 * Problem: Find maximum sum of any path in binary tree
 * A path can start and end at any nodes
 * 
 * Optimal Approach: Recursive with global maximum tracking
 * Time: O(n), Space: O(h)
 */
public class BinaryTreeMaximumPathSum {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    private int maxSum;
    
    // Main solution
    public int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        maxPathSumHelper(root);
        return maxSum;
    }
    
    /**
     * Returns maximum sum of path starting from current node going down
     * Updates global maxSum with maximum path sum passing through current node
     */
    private int maxPathSumHelper(TreeNode node) {
        if (node == null) {
            return 0;
        }
        
        // Get maximum sum from left and right subtrees
        // Use Math.max with 0 to ignore negative paths
        int leftSum = Math.max(0, maxPathSumHelper(node.left));
        int rightSum = Math.max(0, maxPathSumHelper(node.right));
        
        // Maximum path sum passing through current node
        int currentMaxPath = node.val + leftSum + rightSum;
        
        // Update global maximum
        maxSum = Math.max(maxSum, currentMaxPath);
        
        // Return maximum sum of path starting from current node
        // Can only choose one direction (left or right) to continue upward
        return node.val + Math.max(leftSum, rightSum);
    }
    
    // Alternative implementation with clearer variable names
    public int maxPathSumAlternative(TreeNode root) {
        int[] globalMax = {Integer.MIN_VALUE};
        maxGainFromNode(root, globalMax);
        return globalMax[1];
    }
    
    private int maxGainFromNode(TreeNode node, int[] globalMax) {
        if (node == null) return 0;
        
        // Maximum gain from left and right subtrees
        int leftGain = Math.max(0, maxGainFromNode(node.left, globalMax));
        int rightGain = Math.max(0, maxGainFromNode(node.right, globalMax));
        
        // Path sum if we include current node as the "peak" of the path
        int pathThroughNode = node.val + leftGain + rightGain;
        
        // Update global maximum
        globalMax[1] = Math.max(globalMax[1], pathThroughNode);
        
        // Return max gain if we start path from current node
        return node.val + Math.max(leftGain, rightGain);
    }
    
    // Version that also returns the actual path
    public class PathResult {
        int maxSum;
        List<Integer> path;
        
        PathResult(int sum, List<Integer> path) {
            this.maxSum = sum;
            this.path = path;
        }
    }
    
    private PathResult maxPathWithPath;
    
    public PathResult maxPathSumWithPath(TreeNode root) {
        maxPathWithPath = new PathResult(Integer.MIN_VALUE, new ArrayList<>());
        maxPathSumWithPathHelper(root);
        return maxPathWithPath;
    }
    
    private int maxPathSumWithPathHelper(TreeNode node) {
        if (node == null) return 0;
        
        int leftSum = Math.max(0, maxPathSumWithPathHelper(node.left));
        int rightSum = Math.max(0, maxPathSumWithPathHelper(node.right));
        
        int currentMaxPath = node.val + leftSum + rightSum;
        
        if
 (currentMaxPath > maxPathWithPath.maxSum) {
            maxPathWithPath.maxSum = currentMaxPath;
            // Build path (simplified - would need more complex logic for actual path)
            maxPathWithPath.path = Arrays.asList(node.val);
        }
        
        return node.val + Math.max(leftSum, rightSum);
    }
}

/*
Algorithm Explanation:

Key Insights:
1. A path can start and end at any nodes (doesn't have to go through root)
2. For each node, we consider it as the "peak" of a potential path
3. Path through a node = node.val + max_left_path + max_right_path
4. But when returning to parent, we can only choose one direction

Two concepts to track:
1. Maximum path sum passing THROUGH current node (for global answer)
2. Maximum path sum starting FROM current node going down (for parent)

Example tree:
      1
     / \
    2   3
   / \
  4   5

Step-by-step execution:

maxPathSumHelper(4):
  leftSum = 0, rightSum = 0
  currentMaxPath = 4 + 0 + 0 = 4
  maxSum = max(MIN_VALUE, 4) = 4
  return 4 + max(0, 0) = 4

maxPathSumHelper(5):
  leftSum = 0, rightSum = 0
  currentMaxPath = 5 + 0 + 0 = 5
  maxSum = max(4, 5) = 5
  return 5 + max(0, 0) = 5

maxPathSumHelper(2):
  leftSum = max(0, 4) = 4
  rightSum = max(0, 5) = 5
  currentMaxPath = 2 + 4 + 5 = 11
  maxSum = max(5, 11) = 11
  return 2 + max(4, 5) = 7

maxPathSumHelper(3):
  leftSum = 0, rightSum = 0
  currentMaxPath = 3 + 0 + 0 = 3
  maxSum = max(11, 3) = 11
  return 3 + max(0, 0) = 3

maxPathSumHelper(1):
  leftSum = max(0, 7) = 7
  rightSum = max(0, 3) = 3
  currentMaxPath = 1 + 7 + 3 = 11
  maxSum = max(11, 11) = 11
  return 1 + max(7, 3) = 8

Final answer: 11 (path: 4 -> 2 -> 5)

Key Points:

1. Math.max(0, childSum):
   - Ignore negative paths
   - If subtree gives negative sum, better to not include it

2. Two different return values:
   - currentMaxPath: path through current node (both children)
   - return value: path from current node (one child only)

3. Global variable:
   - Needed because the maximum path might not include root
   - Each node updates global maximum with its "through-node" path

Edge Cases:
- Single node: return node value
- All negative values: return least negative value
- Empty tree: handle appropriately
- Linear tree: becomes simple path sum

Time Complexity: O(n) - visit each node once
Space Complexity: O(h) - recursion stack depth

Common Mistakes:
1. Not using Math.max(0, childSum) to ignore negative paths
2. Confusing the two concepts: path through node vs path from node
3. Not handling negative values correctly
4. Forgetting to update global maximum
5. Returning wrong value (should return single-direction path sum)

Variations:
- Path must go through root: simpler, just leftMax + root + rightMax
- Path must start/end at leaf: modify base case
- Return actual path nodes: need additional tracking
- Maximum product path: similar approach with multiplication

*/
```

### 29. Binary Tree Level Order Traversal

```java
import java.util.*;

/**
 * Problem: Return level order traversal of binary tree (BFS)
 * 
 * Multiple approaches: Queue-based, Recursive
 */
public class BinaryTreeLevelOrderTraversal {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Iterative BFS with Queue - Most common
    // Time: O(n), Space: O(w) where w is maximum width
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            // Process all nodes at current level
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                // Add children for next level
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            result.add(currentLevel);
        }
        
        return result;
    }
    
    // Approach 2: Recursive DFS with level tracking
    // Time: O(n), Space: O(h) where h is height
    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        levelOrderHelper(root, 0, result);
        return result;
    }
    
    private void levelOrderHelper(TreeNode node, int level, List<List<Integer>> result) {
        if (node == null) return;
        
        // Create new level list if needed
        if (level >= result.size()) {
            result.add(new ArrayList<>());
        }
        
        // Add current node to its level
        result.get(level).add(node.val);
        
        // Recursively process children
        levelOrderHelper(node.left, level + 1, result);
        levelOrderHelper(node.right, level + 1, result);
    }
    
    // Approach 3: Two Queue approach (alternative BFS)
    // Time: O(n), Space: O(w)
    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> currentLevel = new LinkedList<>();
        currentLevel.offer(root);
        
        while (!currentLevel.isEmpty()) {
            Queue<TreeNode> nextLevel = new LinkedList<>();
            List<Integer> levelValues = new ArrayList<>();
            
            while (!currentLevel.isEmpty()) {
                TreeNode node = currentLevel.poll();
                levelValues.add(node.val);
                
                if (node.left != null) {
                    nextLevel.offer(node.left);
                }
                if (node.right != null) {
                    nextLevel.offer(node.right);
                }
            }
            
            result.add(levelValues);
            currentLevel = nextLevel;
        }
        
        return result;
    }
    
    // Approach 4: Using null as level separator
    // Time: O(n), Space: O(w)
    public List<List<Integer>> levelOrder4(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        queue.offer(null); // Level separator
        
        List<Integer> currentLevel = new ArrayList<>();
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            
            if (node == null) {
                // End of current level
                result.add(new ArrayList<>(currentLevel));
                currentLevel.clear();
                
                // Add separator for next level if queue not empty
                if (!queue.isEmpty()) {
                    queue.offer(null);
                }
            } else {
                currentLevel.add(node.val);
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        
        return result;
    }
    
    // Approach 5: Level order with node depth tracking
    // Time: O(n), Space: O(w)
    public List<List<Integer>> levelOrder5(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> levelQueue = new LinkedList<>();
        
        nodeQueue.offer(root);
        levelQueue.offer(0);
        
        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.poll();
            int level = levelQueue.poll();
            
            // Ensure result has enough levels
            while (result.size() <= level) {
                result.add(new ArrayList<>());
            }
            
            result.get(level).add(node.val);
            
            if (node.left != null) {
                nodeQueue.offer(node.left);
                levelQueue.offer(level + 1);
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
                levelQueue.offer(level + 1);
            }
        }
        
        return result;
    }
}

/*
Algorithm Explanation:

Level order traversal visits nodes level by level from
 left to right.

Example tree:
    3
   / \
  9   20
     /  \
    15   7

Expected output: [[3], [9, 20], [15, 7]]

Approach 1 (Standard BFS):
1. Use queue to store nodes
2. Process all nodes at current level before moving to next
3. Track level size to know when level ends

Step by step:
Initial: queue = [3]
Level 0: process 3, add [9, 20] to queue, result = [[3]]
Level 1: process 9, 20, add [15, 7] to queue, result = [[3], [9, 20]]
Level 2: process 15, 7, result = [[3], [9, 20], [15, 7]]

Approach 2 (Recursive DFS):
1. Use recursion with level parameter
2. Create new level lists as needed
3. Add nodes to appropriate level list

Approach 3 (Two Queues):
1. Use separate queues for current and next level
2. Swap queues after processing each level
3. Clear and intuitive separation

Approach 4 (Null Separator):
1. Use null as marker for end of level
2. Add null after processing each level
3. Simple but requires careful null handling

Approach 5 (Level Tracking):
1. Store both node and its level in queues
2. Explicit level tracking
3. More memory but very clear logic

Approach Comparison:

1. Standard BFS (Approach 1):
   - Most commonly used and efficient
   - Clear level separation logic
   - Optimal space usage

2. Recursive DFS (Approach 2):
   - Elegant recursive solution
   - Uses call stack instead of queue
   - Good for understanding recursion

3. Two Queues (Approach 3):
   - Very intuitive level separation
   - Uses more space (two queues)
   - Easy to understand and debug

4. Null Separator (Approach 4):
   - Creative use of null markers
   - Requires careful null handling
   - Can be error-prone

5. Level Tracking (Approach 5):
   - Explicit level information
   - Uses more space for level queue
   - Very clear what's happening

Time Complexity: O(n) for all approaches - visit each node once
Space Complexity:
- BFS approaches: O(w) where w is maximum width of tree
- Recursive: O(h) where h is height of tree
- Worst case: O(n) for completely unbalanced tree

Key Insights:
1. BFS naturally processes nodes level by level
2. Queue size at start of iteration = number of nodes in current level
3. All children of current level become next level
4. Recursive approach can also achieve level order with proper level tracking

Edge Cases:
- Empty tree: return empty list
- Single node: return [[node.val]]
- Linear tree: each level has one node
- Complete binary tree: levels have 1, 2, 4, 8, ... nodes

Applications:
- Tree visualization
- Finding nodes at specific level
- Level-wise processing
- Building tree from level order input

Common Mistakes:
1. Not tracking level boundaries correctly
2. Adding children to wrong level
3. Forgetting to handle null nodes
4. Incorrect queue size calculation
5. Not initializing level lists properly

Variations:
- Right to left level order: process right child first
- Zigzag level order: alternate left-to-right and right-to-left
- Level order bottom-up: reverse the
 result
- Only leaf nodes at each level: filter non-leaf nodes
*/
```

### 30. Serialize and Deserialize Binary Tree

```java
import java.util.*;

/**
 * Problem: Serialize binary tree to string and deserialize back to tree
 * 
 * Multiple approaches: Preorder, Level order, Postorder
 */
public class SerializeAndDeserializeBinaryTree {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Preorder traversal with recursion - Most intuitive
    public class Codec1 {
        private static final String NULL_MARKER = "null";
        private static final String DELIMITER = ",";
        
        // Encodes a tree to a single string
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeHelper(root, sb);
            return sb.toString();
        }
        
        private void serializeHelper(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append(NULL_MARKER).append(DELIMITER);
                return;
            }
            
            sb.append(node.val).append(DELIMITER);
            serializeHelper(node.left, sb);
            serializeHelper(node.right, sb);
        }
        
        // Decodes your encoded data to tree
        public TreeNode deserialize(String data) {
            Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(DELIMITER)));
            return deserializeHelper(queue);
        }
        
        private TreeNode deserializeHelper(Queue<String> queue) {
            String val = queue.poll();
            
            if (NULL_MARKER.equals(val)) {
                return null;
            }
            
            TreeNode node = new TreeNode(Integer.parseInt(val));
            node.left = deserializeHelper(queue);
            node.right = deserializeHelper(queue);
            
            return node;
        }
    }
    
    // Approach 2: Level order traversal (BFS) - More intuitive for some
    public class Codec2 {
        private static final String NULL_MARKER = "null";
        private static final String DELIMITER = ",";
        
        public String serialize(TreeNode root) {
            if (root == null) return "";
            
            StringBuilder sb = new StringBuilder();
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                
                if (node == null) {
                    sb.append(NULL_MARKER).append(DELIMITER);
                } else {
                    sb.append(node.val).append(DELIMITER);
                    queue.offer(node.left);
                    queue.offer(node.right);
                }
            }
            
            return sb.toString();
        }
        
        public TreeNode deserialize(String data) {
            if (data.isEmpty()) return null;
            
            String[] values = data.split(DELIMITER);
            TreeNode root = new TreeNode(Integer.parseInt(values[1]));
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            int index = 1;
            while (!queue.isEmpty() && index < values.length) {
                TreeNode node = queue.poll();
                
                // Process left child
                if (!NULL_MARKER.equals(values[index])) {
                    node.left = new TreeNode(Integer.parseInt(values[index]));
                    queue.offer(node.left);
                }
                index++;
                
                // Process right child
                if (index < values.length && !NULL_MARKER.equals(values[index])) {
                    node.right = new TreeNode(Integer.parseInt(values[index]));
                    queue.offer(node.right);
                }
                index++;
            }
            
            return root;
        }
    }
    
    // Approach 3: Postorder traversal - Alternative recursive approach
    public class Codec3 {
        private static final String NULL_MARKER = "null";
        private static final String DELIMITER = ",";
        
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializePostorder(root, sb);
            return sb.toString();
        }
        
        private void serializePostorder(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append(NULL_MARKER).append(DELIMITER);
                return;
            }
            
            serializePostorder(node.left, sb);
            serializePostorder(node.right, sb);
            sb.append(node.val).append(DELIMITER);
        }
        
        public TreeNode deserialize(String data) {
            Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(DELIMITER)));
            return deserializePostorder(queue);
        }
        
        private TreeNode deserializePostorder(Queue<String> queue) {
            String val = queue.poll();
            
            if (NULL_MARKER.equals(val)) {
                return null;
            }
            
            TreeNode node = new TreeNode(Integer.parseInt(val));
            node.right = deserializePostorder(queue);
            node.left = deserializePostorder(queue);
            
            return node;
        }
    }
    
    // Approach 4: Compact representation (no null markers for leaves)
    public class Codec4 {
        private static final String DELIMITER = ",";
        
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeCompact(root, sb);
            return sb.toString();
        }
        
        private void serializeCompact(TreeNode node, StringBuilder sb) {
            if (node == null) return;
            
            sb.append(node.val).append(DELIMITER);
            
            // Add markers for structure
            if (node.left != null ||
 node.right != null) {
                sb.append("(");
                serializeCompact(node.left, sb);
                sb.append(")(");
                serializeCompact(node.right, sb);
                sb.append(")");
            }
        }
        
        // Deserialize would be more complex for this approach
        // Included for completeness but not fully implemented
        public TreeNode deserialize(String data) {
            // Complex parsing required for this format
            // Would need proper parentheses parsing
            return null;
        }
    }
    
    // Approach 5: Using indices (array representation)
    public class Codec5 {
        private static final String NULL_MARKER = "null";
        private static final String DELIMITER = ",";
        
        public String serialize(TreeNode root) {
            List<String> result = new ArrayList<>();
            serializeWithIndex(root, 0, result);
            return String.join(DELIMITER, result);
        }
        
        private void serializeWithIndex(TreeNode node, int index, List<String> result) {
            if (node == null) return;
            
            // Ensure list is large enough
            while (result.size() <= index) {
                result.add(NULL_MARKER);
            }
            
            result.set(index, String.valueOf(node.val));
            serializeWithIndex(node.left, 2 * index + 1, result);
            serializeWithIndex(node.right, 2 * index + 2, result);
        }
        
        public TreeNode deserialize(String data) {
            if (data.isEmpty()) return null;
            
            String[] values = data.split(DELIMITER);
            return deserializeWithIndex(values, 0);
        }
        
        private TreeNode deserializeWithIndex(String[] values, int index) {
            if (index >= values.length || NULL_MARKER.equals(values[index])) {
                return null;
            }
            
            TreeNode node = new TreeNode(Integer.parseInt(values[index]));
            node.left = deserializeWithIndex(values, 2 * index + 1);
            node.right = deserializeWithIndex(values, 2 * index + 2);
            
            return node;
        }
    }
}

/*
Algorithm Explanation:

Serialization converts tree structure to string format.
Deserialization reconstructs tree from string.

Example tree:
    1
   / \
  2   3
     / \
    4   5

Approach 1 (Preorder): "1,2,null,null,3,4,null,null,5,null,null,"
- Visit root, then left subtree, then right subtree
- Include null markers to preserve structure

Approach 2 (Level order): "1,2,3,null,null,4,5,null,null,null,null,"
- Visit nodes level by level
- Include null markers for missing children

Approach 3 (Postorder): "null,null,2,null,null,4,null,null,5,3,1,"
- Visit left subtree, right subtree, then root
- Reconstruction works backwards

Approach Comparison:

1. Preorder (Most Common):
   - Natural recursive structure
   - Easy to implement and understand
   - Efficient serialization and deserialization

2. Level Order:
   - Intuitive for visualization
   - Good for breadth-first processing
   - Slightly more complex deserialization

3. Postorder:
   - Alternative recursive approach
   - Requires reverse processing for deserialization
   - Less commonly used

4. Compact:
   - Saves space by avoiding unnecessary null markers
   - More complex parsing required
   - Good for space-constrained scenarios

5. Index-based:
   - Uses array representation concept
   - Can be sparse for unbalanced trees
   - Good for complete/nearly complete trees

Key Design Decisions:

1. Null Representation:
   - Use special marker (e.g., "null", "#", "X")
   - Must be distinguishable from valid node values

2. Delimiter:
   - Separate values (comma, space, etc.)
   - Must not conflict with node values

3. Traversal Order:
   - Preorder: most natural for recursion
   - Level order: most intuitive for humans
   - Postorder: alternative recursive approach

Implementation Details:

Serialization:
- Traverse tree in chosen order
- Convert node
 values to strings
- Add null markers for missing nodes
- Join with delimiters

Deserialization:
- Split string by delimiter
- Reconstruct tree following same order
- Handle null markers appropriately
- Use queue/recursion for processing

Time Complexity: O(n) for both operations
Space Complexity: O(n) for serialized string, O(h) for recursion stack

Edge Cases:
- Empty tree: return empty string or special marker
- Single node: handle appropriately
- Large values: ensure no delimiter conflicts
- Negative values: handle sign correctly

Applications:
- Tree persistence to file/database
- Network transmission of trees
- Tree comparison and hashing
- Debugging and visualization

Common Mistakes:
1. Not handling null nodes properly
2. Delimiter conflicts with node values
3. Incorrect reconstruction order
4. Not preserving tree structure
5. Memory leaks in deserialization

Best Practices:
- Use preorder for simplicity
- Choose delimiters carefully
- Handle edge cases explicitly
- Add validation for malformed input
- Consider space efficiency for large trees
*/
```

### 31. Subtree of Another Tree

```java
/**
 * Problem: Check if tree s is a subtree of tree t
 * A subtree must include all descendants of a node
 * 
 * Multiple approaches: Recursive, Serialization
 */
public class SubtreeOfAnotherTree {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive - Check each node as potential subtree root
    // Time: O(m*n) worst case, Space: O(max(m,n))
    public boolean isSubtree1(TreeNode root, TreeNode subRoot) {
        if (subRoot == null) return true;  // Empty tree is subtree of any tree
        if (root == null) return false;    // Non-empty tree can't be subtree of empty tree
        
        // Check if subtree starts at current root, or in left/right subtrees
        return isSameTree(root, subRoot) || 
               isSubtree1(root.left, subRoot) || 
               isSubtree1(root.right, subRoot);
    }
    
    // Helper method to check if two trees are identical
    private boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        
        return p.val == q.val && 
               isSameTree(p.left, q.left) && 
               isSameTree(p.right, q.right);
    }
    
    // Approach 2: Serialization - Convert trees to strings and check substring
    // Time: O(m+n), Space: O(m+n)
    public boolean isSubtree2(TreeNode root, TreeNode subRoot) {
        String rootStr = serialize(root);
        String subRootStr = serialize(subRoot);
        
        return rootStr.contains(subRootStr);
    }
    
    private String serialize(TreeNode node) {
        if (node == null) {
            return "null";
        }
        
        // Use unique delimiters to avoid false matches
        return "#" + node.val + " " + serialize(node.left) + " " + serialize(node.right);
    }
    
    // Approach 3: Optimized recursive with early termination
    // Time: O(m*n) worst case, but better average case
    public boolean isSubtree3(TreeNode root, TreeNode subRoot) {
        if (subRoot == null) return true;
        if (root == null) return false;
        
        // Early termination: if current subtree is smaller than target, skip
        if (countNodes(root) < countNodes(subRoot)) {
            return false;
        }
        
        return isSameTree(root, subRoot) || 
               isSubtree3(root.left, subRoot) || 
               isSub
tree3(root.right, subRoot);
    }
    
    private int countNodes(TreeNode node) {
        if (node == null) return 0;
        return 1 + countNodes(node.left) + countNodes(node.right);
    }
    
    // Approach 4: Hash-based comparison
    // Time: O(m+n), Space: O(m+n)
    public boolean isSubtree4(TreeNode root, TreeNode subRoot) {
        Map<String, Integer> rootHashes = new HashMap<>();
        Map<String, Integer> subHashes = new HashMap<>();
        
        String subRootHash = getHash(subRoot, subHashes);
        getHash(root, rootHashes);
        
        return rootHashes.containsKey(subRootHash);
    }
    
    private String getHash(TreeNode node, Map<String, Integer> hashes) {
        if (node == null) return "null";
        
        String left = getHash(node.left, hashes);
        String right = getHash(node.right, hashes);
        String current = node.val + "," + left + "," + right;
        
        hashes.put(current, hashes.getOrDefault(current, 0) + 1);
        return current;
    }
    
    // Approach 5: Merkle tree approach with structural hashing
    // Time: O(m+n), Space: O(m+n)
    public boolean isSubtree5(TreeNode root, TreeNode subRoot) {
        Map<TreeNode, String> memo = new HashMap<>();
        String subRootStructure = getStructuralHash(subRoot, memo);
        return findStructure(root, subRootStructure, memo);
    }
    
    private String getStructuralHash(TreeNode node, Map<TreeNode, String> memo) {
        if (node == null) return "null";
        if (memo.containsKey(node)) return memo.get(node);
        
        String left = getStructuralHash(node.left, memo);
        String right = getStructuralHash(node.right, memo);
        String hash = "(" + left + ")" + node.val + "(" + right + ")";
        
        memo.put(node, hash);
        return hash;
    }
    
    private boolean findStructure(TreeNode node, String target, Map<TreeNode, String> memo) {
        if (node == null) return false;
        
        if (getStructuralHash(node, memo).equals(target)) {
            return true;
        }
        
        return findStructure(node.left, target, memo) || 
               findStructure(node.right, target, memo);
    }
}

/*
Algorithm Explanation:

A subtree means:
1. The subtree must be rooted at some node in the main tree
2. It must include ALL descendants of that node
3. Structure and values must match exactly

Example:
Main tree:     Subtree:
    3             4
   / \           / \
  4   5         1   2
 / \
1   2

The subtree rooted at node 4 in main tree matches the given subtree.

Approach 1 (Recursive):
For each node in main tree:
1. Check if subtree starting at this node matches target
2. If not, recursively check left and right children
3. Use helper function to compare two trees for equality

Time complexity analysis:
- For each of m nodes in main tree, we might do O(n) comparison
- Worst case: O(m*n) when we check every node
- Best case: O(m) when subtree is found quickly

Approach 2 (Serialization):
1. Convert both trees to string representation
2. Check if subtree string is substring of main tree string
3. Use unique delimiters to avoid false positives

Example serialization:
Main: "#3 #4 #1 null null #2 null null #5 null null"
Sub:  "#4 #1 null null #2 null null"

The substring check will find the match.

Approach 3 (Optimized Recursive):
Add early termination conditions:
1. Count nodes in current subtree
2. If fewer nodes than target, skip this subtree
3. Reduces unnecessary comparisons

Approach 4 (Hash-based):
1. Generate hash for each subtree structure
2. Store hashes in map
3. Check if target subtree hash exists in main tree hashes

Approach 5 (Merkle Tree):
1. Create structural hash for each node
2. Hash includes structure and values
3. Use memoization to avoid recomputation


Approach Comparison:

1. Recursive (Approach 1):
   - Most intuitive and commonly used
   - Easy to understand and implement
   - O(m*n) worst case but often better in practice

2. Serialization (Approach 2):
   - Linear time complexity O(m+n)
   - Simple concept but requires careful delimiter choice
   - Can have false positives if not implemented carefully

3. Optimized Recursive (Approach 3):
   - Better average case performance
   - Early termination reduces unnecessary work
   - Still O(m*n) worst case

4. Hash-based (Approach 4):
   - Linear time with good hash function
   - Risk of hash collisions
   - More complex implementation

5. Merkle Tree (Approach 5):
   - Efficient with memoization
   - Good for multiple queries
   - Complex implementation

Key Insights:
1. Subtree must be exact match including structure
2. Need to check every possible starting position
3. Early termination can significantly improve performance
4. Serialization converts tree problem to string problem

Edge Cases:
- Empty subtree: always true (empty set is subset of any set)
- Empty main tree with non-empty subtree: false
- Single node trees: check value equality
- Identical trees: true
- Subtree larger than main tree: false

Common Mistakes:
1. Not checking exact structure match
2. Forgetting to handle null nodes
3. Incorrect serialization causing false positives
4. Not considering all possible starting positions
5. Confusing subtree with subsequence

Optimization Techniques:
1. Early termination based on size
2. Memoization of structural hashes
3. Efficient string matching algorithms
4. Pruning based on value ranges
5. Using rolling hash for better performance

When to use each approach:
- Interview/Simple case: Recursive (Approach 1)
- Performance critical: Serialization (Approach 2)
- Multiple queries: Hash-based (Approach 4)
- Educational: All approaches show different techniques
*/
```

### 32. Construct Binary Tree from Preorder and Inorder Traversal

```java
import java.util.*;

/**
 * Problem: Build binary tree from preorder and inorder traversal arrays
 * 
 * Key insight: Preorder gives root, inorder gives left/right subtree split
 */
public class ConstructBinaryTreeFromPreorderAndInorder {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive with HashMap for O(1) inorder lookups
    // Time: O(n), Space: O(n)
    public TreeNode buildTree1(int[] preorder, int[] inorder) {
        // Build map for O(1) inorder index lookup
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        return buildTreeHelper(preorder, 0, preorder.length - 1,
                              inorder, 0, inorder.length - 1, inorderMap);
    }
    
    private TreeNode buildTreeHelper(int[] preorder, int preStart, int preEnd,
                                   int[] inorder, int inStart, int inEnd,
                                   Map<Integer, Integer> inorderMap) {
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }
        
        // Root is first element in preorder
        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);
        
        // Find root position in inorder
        int rootIndex = inorderMap.get(rootVal);
        int leftSubtreeSize = rootIndex - inStart;
        
        // Build left subtree
        root.left = buildTreeHelper(preorder, preStart + 1, preStart + leftSubtreeSize,
                                   inorder, inStart, rootIndex - 1, inorderMap);
        
        // Build right subtree
        root.right = buildTreeHelper(preorder, preStart + leftSubtreeSize + 1, preEnd,
                                    inorder, rootIndex + 1, inEnd, inorderMap);
        
        return root;
    }
    
    // Approach 2: Using global index for preorder
    // Time: O(n), Space: O(n)
    private int
 preorderIndex = 0;
    
    public TreeNode buildTree2(int[] preorder, int[] inorder) {
        preorderIndex = 0;
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        return buildTreeHelper2(preorder, inorder, 0, inorder.length - 1, inorderMap);
    }
    
    private TreeNode buildTreeHelper2(int[] preorder, int[] inorder, 
                                     int inStart, int inEnd,
                                     Map<Integer, Integer> inorderMap) {
        if (inStart > inEnd) {
            return null;
        }
        
        // Current root from preorder
        int rootVal = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootVal);
        
        // Find root in inorder
        int rootIndex = inorderMap.get(rootVal);
        
        // Build left subtree first (preorder: root -> left -> right)
        root.left = buildTreeHelper2(preorder, inorder, inStart, rootIndex - 1, inorderMap);
        root.right = buildTreeHelper2(preorder, inorder, rootIndex + 1, inEnd, inorderMap);
        
        return root;
    }
    
    // Approach 3: Without HashMap (less efficient but educational)
    // Time: O(n²), Space: O(n)
    public TreeNode buildTree3(int[] preorder, int[] inorder) {
        return buildTreeHelper3(preorder, 0, preorder.length - 1,
                               inorder, 0, inorder.length - 1);
    }
    
    private TreeNode buildTreeHelper3(int[] preorder, int preStart, int preEnd,
                                     int[] inorder, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }
        
        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);
        
        // Find root in inorder array (O(n) search)
        int rootIndex = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == rootVal) {
                rootIndex = i;
                break;
            }
        }
        
        int leftSubtreeSize = rootIndex - inStart;
        
        root.left = buildTreeHelper3(preorder, preStart + 1, preStart + leftSubtreeSize,
                                    inorder, inStart, rootIndex - 1);
        root.right = buildTreeHelper3(preorder, preStart + leftSubtreeSize + 1, preEnd,
                                     inorder, rootIndex + 1, inEnd);
        
        return root;
    }
    
    // Approach 4: Iterative using stack
    // Time: O(n), Space: O(n)
    public TreeNode buildTree4(int[] preorder, int[] inorder) {
        if (preorder.length == 0) return null;
        
        TreeNode root = new TreeNode(preorder[1]);
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        
        int inorderIndex = 0;
        
        for (int i = 1; i < preorder.length; i++) {
            TreeNode node = new TreeNode(preorder[i]);
            TreeNode parent = null;
            
            // Find the correct parent for current node
            while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]) {
                parent = stack.pop();
                inorderIndex++;
            }
            
            if (parent != null) {
                // Current node is right child of parent
                parent.right = node;
            } else {
                // Current node is left child of stack top
                stack.peek().left = node;
            }
            
            stack.push(node);
        }
        
        return root;
    }
}

/*
Algorithm Explanation:

Key insights:
1. Preorder: Root -> Left -> Right (first element is always root)
2. Inorder: Left -> Root -> Right (root splits left and right subtrees)
3. Use preorder to identify roots, inorder to determine subtree boundaries

Example:
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]

Step-by-step construction:

1. Root = 3 (first in preorder)
   Inorder: [9] | 3 | [15,20,7]
   Left subtree: [9], Right subtree: [15,20,7]

2. Left subtree:
   preorder = [9], inorder = [9]
   Root = 9, no children

3. Right subtree:
   preorder = [20,15,7], inorder = [15,20,7]
   Root = 20 (first in preorder)
   Inorder: [15] | 20 | [7]
   Left: [15], Right: [7]

4. Continue recursively...

Final tree:
    3
   / \
  9   20
     /  
\
    15   7

Approach 1 (HashMap optimization):
- Use HashMap to find root position in inorder in O(1)
- Calculate subtree sizes to determine array boundaries
- Recursively build left and right subtrees

Approach 2 (Global index):
- Use global preorder index instead of calculating ranges
- Simpler index management
- Same time complexity but cleaner code

Approach 3 (No HashMap):
- Linear search to find root in inorder
- O(n²) time complexity due to repeated searches
- Educational but not optimal

Approach 4 (Iterative):
- Use stack to simulate recursion
- More complex but avoids recursion overhead
- Good for very deep trees

Key Calculations:

For each recursive call:
1. Root = preorder[preStart]
2. Find rootIndex in inorder
3. leftSubtreeSize = rootIndex - inStart
4. Left subtree: preorder[preStart+1 : preStart+leftSubtreeSize]
                 inorder[inStart : rootIndex-1]
5. Right subtree: preorder[preStart+leftSubtreeSize+1 : preEnd]
                  inorder[rootIndex+1 : inEnd]

Time Complexity:
- Approach 1,2: O(n) - each node processed once, O(1) lookups
- Approach 3: O(n²) - O(n) search for each of n nodes
- Approach 4: O(n) - each node pushed/popped once

Space Complexity:
- All approaches: O(n) for recursion stack or explicit stack
- HashMap: additional O(n) space

Edge Cases:
- Empty arrays: return null
- Single element: return single node
- Linear tree: works correctly
- Complete binary tree: optimal case

Constraints and Assumptions:
- All values are unique (required for unambiguous reconstruction)
- Both arrays have same length
- Both arrays represent same tree

Common Mistakes:
1. Incorrect boundary calculations
2. Not handling empty subarrays
3. Off-by-one errors in index calculations
4. Forgetting to build HashMap for optimization
5. Incorrect left/right subtree size calculation

Variations:
- Postorder + Inorder: similar approach, root is last in postorder
- Preorder + Postorder: possible only with full binary tree
- Level order + Inorder: more complex but possible

Applications:
- Tree serialization/deserialization
- Reconstructing trees from traversal data
- Compiler construction (AST building)
- Database index reconstruction
*/
```

### 33. Validate Binary Search Tree

```java
/**
 * Problem: Check if binary tree is a valid BST
 * 
 * Key insight: Each node must be within a valid range based on ancestors
 */
public class ValidateBinarySearchTree {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive with min/max bounds - Most efficient
    // Time: O(n), Space: O(h)
    public boolean isValidBST1(TreeNode root) {
        return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    
    private boolean validate(TreeNode node, long minVal, long maxVal) {
        if (node == null) {
            return true;
        }
        
        // Check if current node violates BST property
        if (node.val <= minVal || node.val >= maxVal) {
            return false;
        }
        
        // Recursively validate left and right subtrees with updated bounds
        return validate(node.left, minVal, node.
val) && 
               validate(node.right, node.val, maxVal);
    }
    
    // Approach 2: Inorder traversal - Should be sorted for valid BST
    // Time: O(n), Space: O(h)
    private Integer prevValue = null;
    
    public boolean isValidBST2(TreeNode root) {
        prevValue = null;
        return inorderCheck(root);
    }
    
    private boolean inorderCheck(TreeNode node) {
        if (node == null) {
            return true;
        }
        
        // Check left subtree
        if (!inorderCheck(node.left)) {
            return false;
        }
        
        // Check current node
        if (prevValue != null && node.val <= prevValue) {
            return false;
        }
        prevValue = node.val;
        
        // Check right subtree
        return inorderCheck(node.right);
    }
    
    // Approach 3: Iterative inorder traversal
    // Time: O(n), Space: O(h)
    public boolean isValidBST3(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        Integer prev = null;
        TreeNode current = root;
        
        while (current != null || !stack.isEmpty()) {
            // Go to leftmost node
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            
            // Process current node
            current = stack.pop();
            
            // Check BST property
            if (prev != null && current.val <= prev) {
                return false;
            }
            prev = current.val;
            
            // Move to right subtree
            current = current.right;
        }
        
        return true;
    }
    
    // Approach 4: Collect inorder values and check if sorted
    // Time: O(n), Space: O(n)
    public boolean isValidBST4(TreeNode root) {
        List<Integer> inorderValues = new ArrayList<>();
        inorderTraversal(root, inorderValues);
        
        // Check if list is strictly increasing
        for (int i = 1; i < inorderValues.size(); i++) {
            if (inorderValues.get(i) <= inorderValues.get(i - 1)) {
                return false;
            }
        }
        
        return true;
    }
    
    private void inorderTraversal(TreeNode node, List<Integer> values) {
        if (node == null) return;
        
        inorderTraversal(node.left, values);
        values.add(node.val);
        inorderTraversal(node.right, values);
    }
    
    // Approach 5: Using wrapper class to avoid global variables
    // Time: O(n), Space: O(h)
    public boolean isValidBST5(TreeNode root) {
        return isValidBSTHelper(root).isValid;
    }
    
    private class ValidationResult {
        boolean isValid;
        Integer min;
        Integer max;
        
        ValidationResult(boolean isValid, Integer min, Integer max) {
            this.isValid = isValid;
            this.min = min;
            this.max = max;
        }
    }
    
    private ValidationResult isValidBSTHelper(TreeNode node) {
        if (node == null) {
            return new ValidationResult(true, null, null);
        }
        
        ValidationResult left = isValidBSTHelper(node.left);
        ValidationResult right = isValidBSTHelper(node.right);
        
        // Check if subtrees are valid and current node maintains BST property
        boolean isValid = left.isValid && right.isValid &&
                         (left.max == null || left.max < node.val) &&
                         (right.min == null || right.min > node.val);
        
        Integer min = (left.min != null) ? left.min : node.val;
        Integer max = (right.max != null) ? right.max : node.val;
        
        return new ValidationResult(isValid, min, max);
    }
}

/*
Algorithm Explanation:

A valid BST must satisfy:
1. Left subtree contains only nodes with values < root
2. Right subtree contains only nodes with values > root
3. Both left and right subtrees are also valid BSTs
4. All values must be unique (no duplicates)

Common mistake: Only checking immediate children
Invalid example:
    5
   / \
  1   4
     / \
    3   6

Node 3 is in right subtree of 5 but 3 < 5, violating BST property.

Approach 1 (Min/Max Bounds):
Maintain valid range for each node based on ancestors.
- Root can be any value: (-∞, +∞)
- Left child: (min, parent_val)
- Right child: (parent_val, max)

Example trace for tree [2,1,3]:
validate(2, -∞, +∞): 2 is in range
  validate(1, -∞, 2): 1 is in range
    validate(null): true
    validate(null): true
  validate(3, 2, +∞): 3 is in range
    validate(null): true
    validate(null): true
All checks
 pass → valid BST

Approach 2 (Inorder Traversal):
Inorder traversal of BST should give sorted sequence.
Use previous value to check if current value is greater.

Example: [2,1,3] → inorder: 1,2,3 (sorted) → valid BST
Example: [5,1,4,null,null,3,6] → inorder: 1,5,3,4,6 (not sorted) → invalid

Approach 3 (Iterative Inorder):
Same logic as approach 2 but using explicit stack.
Avoids recursion overhead and global variables.

Approach 4 (Collect and Check):
Store all inorder values and check if array is sorted.
Simple but uses O(n) extra space.

Approach 5 (Bottom-up Validation):
Return min/max values from subtrees to validate parent.
More complex but avoids global state.

Approach Comparison:

1. Min/Max Bounds:
   - Most intuitive and efficient
   - Clear logic and easy to understand
   - Handles edge cases well

2. Inorder Traversal:
   - Leverages BST property directly
   - Elegant solution
   - Requires careful state management

3. Iterative Inorder:
   - Avoids recursion overhead
   - Good for very deep trees
   - More complex implementation

4. Collect and Check:
   - Simple to understand
   - Uses extra space
   - Good for educational purposes

5. Bottom-up:
   - Functional programming style
   - Avoids global variables
   - More complex but clean

Time Complexity: O(n) for all approaches - must visit every node
Space Complexity:
- Approaches 1,2,3,5: O(h) for recursion/stack
- Approach 4: O(n) for storing values

Key Insights:
1. Must check global BST property, not just local parent-child relationships
2. Inorder traversal of BST is sorted
3. Each node has valid range based on ancestors
4. Handle integer overflow with Long.MIN_VALUE/MAX_VALUE

Edge Cases:
- Empty tree: valid BST
- Single node: valid BST
- Duplicate values: invalid BST
- Integer.MIN_VALUE/MAX_VALUE nodes: use Long for bounds
- Linear tree: still valid if properly ordered

Common Mistakes:
1. Only checking immediate parent-child relationships
2. Allowing duplicate values
3. Not handling integer overflow in bounds
4. Incorrect inorder traversal implementation
5. Using wrong comparison operators (< vs <=)

Optimization Notes:
- Approach 1 is generally preferred for interviews
- Use Long for bounds to handle edge cases
- Early termination when invalid node found
- Iterative approach for stack-overflow concerns

Applications:
- Database index validation
- Compiler symbol table verification
- Data structure integrity checking
- Tree-based algorithm preprocessing
*/
```

### 34. Kth Smallest Element in a BST

```java
import java.util.*;

/**
 * Problem: Find kth smallest element in BST (1-indexed)
 * 
 * Multiple approaches leveraging BST properties
 */
public class KthSmallestElementInBST {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Inorder traversal with early termination - Most efficient
    // Time: O(H + k) where H is height, Space: O(H)
    public int kthSmallest1(TreeNode root, int k) {
        int[]
 result = new int[2];
        int[] count = new int[2];
        inorderTraversal(root, k, count, result);
        return result[1];
    }
    
    private void inorderTraversal(TreeNode node, int k, int[] count, int[] result) {
        if (node == null || count[1] >= k) {
            return;
        }
        
        // Traverse left subtree
        inorderTraversal(node.left, k, count, result);
        
        // Process current node
        count[1]++;
        if (count[1] == k) {
            result[1] = node.val;
            return;
        }
        
        // Traverse right subtree
        inorderTraversal(node.right, k, count, result);
    }
    
    // Approach 2: Iterative inorder traversal
    // Time: O(H + k), Space: O(H)
    public int kthSmallest2(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        int count = 0;
        
        while (current != null || !stack.isEmpty()) {
            // Go to leftmost node
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            
            // Process current node
            current = stack.pop();
            count++;
            
            if (count == k) {
                return current.val;
            }
            
            // Move to right subtree
            current = current.right;
        }
        
        return -1; // Should never reach here if k is valid
    }
    
    // Approach 3: Collect all values and sort (if BST property not leveraged)
    // Time: O(n log n), Space: O(n)
    public int kthSmallest3(TreeNode root, int k) {
        List<Integer> values = new ArrayList<>();
        collectValues(root, values);
        Collections.sort(values);
        return values.get(k - 1);
    }
    
    private void collectValues(TreeNode node, List<Integer> values) {
        if (node == null) return;
        
        values.add(node.val);
        collectValues(node.left, values);
        collectValues(node.right, values);
    }
    
    // Approach 4: Inorder traversal collecting all values (leverages BST)
    // Time: O(n), Space: O(n)
    public int kthSmallest4(TreeNode root, int k) {
        List<Integer> inorderValues = new ArrayList<>();
        inorderCollect(root, inorderValues);
        return inorderValues.get(k - 1);
    }
    
    private void inorderCollect(TreeNode node, List<Integer> values) {
        if (node == null) return;
        
        inorderCollect(node.left, values);
        values.add(node.val);
        inorderCollect(node.right, values);
    }
    
    // Approach 5: Binary search approach (if we can modify tree structure)
    // Time: O(H) average, Space: O(1)
    public class TreeNodeWithCount {
        int val;
        int leftCount; // Number of nodes in left subtree
        TreeNodeWithCount left;
        TreeNodeWithCount right;
        
        TreeNodeWithCount(int val) {
            this.val = val;
            this.leftCount = 0;
        }
    }
    
    public int kthSmallestWithCount(TreeNodeWithCount root, int k) {
        if (root == null) return -1;
        
        int leftCount = root.leftCount;
        
        if (k <= leftCount) {
            // kth smallest is in left subtree
            return kthSmallestWithCount(root.left, k);
        } else if (k == leftCount + 1) {
            // Current node is kth smallest
            return root.val;
        } else {
            // kth smallest is in right subtree
            return kthSmallestWithCount(root.right, k - leftCount - 1);
        }
    }
    
    // Approach 6: Morris traversal (constant space)
    // Time: O(n), Space: O(1)
    public int kthSmallest6(TreeNode root, int k) {
        int count = 0;
        TreeNode current = root;
        
        while (current != null) {
            if (current.left == null) {
                // Process current node
                count++;
                if (count == k) {
                    return current.val;
                }
                current = current.right;
            } else {
                // Find inorder predecessor
                TreeNode predecessor = current.left;
                while (predecessor.right != null && predecessor.right != current) {
                    predecessor = predecessor.right;
                }
                
                if (predecessor.right == null) {
                    // Make current as right child of predecessor
                    predecessor.right = current;
                    current = current.left;
                } else {
                    // Revert the changes
                    predecessor.right = null;
                    count++;
                    if (count == k) {
                        return current.val;
                    }
                    current = current.right;
                }
            }
        }
        
        return -1;
    }
}

/*
Algorithm Explanation:

Key insight: Inorder traversal of BST gives elements in sorted order.
So kth smallest = kth element in inorder traversal.

Example BST:
    3
   / \
  1   4
   
\
    2

Inorder: 1, 2, 3, 4
1st smallest = 1, 2nd smallest = 2, etc.

Approach 1 (Recursive Inorder with Early Termination):
- Perform inorder traversal
- Count nodes visited
- Stop when count reaches k
- Most efficient: O(H + k) time

Example trace for k=2:
1. Visit node 1: count=1
2. Visit node 2: count=2, return 2

Approach 2 (Iterative Inorder):
- Use explicit stack for inorder traversal
- Count nodes as we visit them
- Return when count reaches k
- Same efficiency as approach 1

Approach 3 (Collect and Sort):
- Collect all values (any traversal)
- Sort the array
- Return kth element
- Doesn't leverage BST property: O(n log n)

Approach 4 (Inorder Collection):
- Collect values using inorder traversal
- Values are already sorted
- Return kth element
- Better than approach 3: O(n)

Approach 5 (Augmented BST):
- Store count of left subtree nodes in each node
- Use binary search logic
- Most efficient for multiple queries: O(H)

Approach 6 (Morris Traversal):
- Constant space inorder traversal
- Uses threading technique
- Complex but space-optimal: O(1) space

Approach Comparison:

1. Recursive Inorder (Best for single query):
   - Time: O(H + k), Space: O(H)
   - Early termination saves time
   - Most commonly used

2. Iterative Inorder:
   - Time: O(H + k), Space: O(H)
   - Avoids recursion overhead
   - Good for deep trees

3. Collect and Sort:
   - Time: O(n log n), Space: O(n)
   - Doesn't use BST property
   - Inefficient but simple

4. Inorder Collection:
   - Time: O(n), Space: O(n)
   - Uses BST property
   - Good for multiple k queries on same tree

5. Augmented BST:
   - Time: O(H), Space: O(1)
   - Requires tree modification
   - Best for frequent queries

6. Morris Traversal:
   - Time: O(n), Space: O(1)
   - Constant space
   - Complex implementation

Time Complexity Analysis:
- H = height of tree
- Best case (balanced): H = log n
- Worst case (skewed): H = n
- Early termination: stop after visiting k nodes

Space Complexity:
- Recursive: O(H) for call stack
- Iterative: O(H) for explicit stack
- Morris: O(1) constant space

Key Insights:
1. BST inorder traversal gives sorted sequence
2. Early termination when k nodes visited
3. Augmented trees enable O(H) queries
4. Morris traversal achieves O(1) space

Edge Cases:
- k = 1: return minimum element (leftmost)
- k = n: return maximum element (rightmost)
- k > n: invalid input
- Single node tree: return root value
- Empty tree: handle appropriately

Optimization Techniques:
1. Early termination in traversal
2. Augment tree with subtree counts
3. Use iterative approach for deep trees
4. Morris traversal for space constraints

Follow-up Questions:
1. What if BST is modified frequently? → Use augmented BST
2. What if we need kth largest? → Reverse inorder or (n-k+1)th smallest
3. What if k is very large? → Consider right-to-left traversal
4. Multiple k queries? → Collect all values once

Common Mistakes:
1. Not leveraging BST property (using general tree algorithms
)
2. Not implementing early termination
3. Off-by-one errors in counting
4. Incorrect inorder traversal implementation
5. Not handling edge cases properly

Applications:
- Database query optimization (ORDER BY LIMIT)
- Selection algorithms
- Median finding in dynamic datasets
- Ranking systems
*/
```

### 35. Lowest Common Ancestor of a Binary Search Tree

```java
/**
 * Problem: Find lowest common ancestor of two nodes in BST
 * 
 * Key insight: Leverage BST property for efficient solution
 */
public class LowestCommonAncestorOfBST {
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Approach 1: Recursive leveraging BST property - Most elegant
    // Time: O(H) where H is height, Space: O(H) for recursion
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        // Base case
        if (root == null) return null;
        
        // If both nodes are in left subtree
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor1(root.left, p, q);
        }
        
        // If both nodes are in right subtree
        if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor1(root.right, p, q);
        }
        
        // If nodes are on different sides or one of them is root
        // Current root is the LCA
        return root;
    }
    
    // Approach 2: Iterative leveraging BST property - Most efficient
    // Time: O(H), Space: O(1)
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode current = root;
        
        while (current != null) {
            // Both nodes in left subtree
            if (p.val < current.val && q.val < current.val) {
                current = current.left;
            }
            // Both nodes in right subtree
            else if (p.val > current.val && q.val > current.val) {
                current = current.right;
            }
            // Found LCA: nodes are on different sides or one is current
            else {
                return current;
            }
        }
        
        return null; // Should never reach here if p and q exist in tree
    }
    
    // Approach 3: Path-based approach (works for general binary tree too)
    // Time: O(H), Space: O(H)
    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> pathToP = new ArrayList<>();
        List<TreeNode> pathToQ = new ArrayList<>();
        
        findPath(root, p, pathToP);
        findPath(root, q, pathToQ);
        
        // Find last common node in both paths
        TreeNode lca = null;
        int minLength = Math.min(pathToP.size(), pathToQ.size());
        
        for (int i = 0; i < minLength; i++) {
            if (pathToP.get(i) == pathToQ.get(i)) {
                lca = pathToP.get(i);
            } else {
                break;
            }
        }
        
        return lca;
    }
    
    private boolean findPath(TreeNode root, TreeNode target, List<TreeNode> path) {
        if (root == null) return false;
        
        path.add(root);
        
        if (root == target) {
            return true;
        }
        
        // Search in appropriate subtree based on BST property
        boolean found = false;
        if (target.val < root.val) {
            found = findPath(root.left, target, path);
        } else {
            found = findPath(root.right, target, path);
        }
        
        if (!found) {
            path.remove(path.size() - 1); // Backtrack
        }
        
        return found;
    }
    
    // Approach 4: Using parent pointers (if available)
    // Time: O(H), Space: O(H)
    public class TreeNodeWithParent {
        int val;
        TreeNodeWithParent left;
        TreeNodeWithParent right;
        TreeNodeWithParent parent;
        
        TreeNodeWithParent(int val) {
            this.val = val;
        }
    }
    
    public TreeNodeWithParent lowestCommonAncestor4(TreeNodeWithParent p, TreeNodeWithParent q) {
        Set<TreeNodeWithParent> ancestors = new HashSet<>();
        
        // Collect all ancestors of p
        TreeNodeWithParent current = p;
        while (current != null) {
            ancestors.add(current);
            current = current.parent;
        }
        
        // Find first common ancestor starting from q
        current = q;
        while (current != null) {
            if (ancestors.contains(current)) {
                return current;
            }
            current = current.parent;
        }
        
        return null;
    }
    
    // Approach 5: General binary tree solution (doesn't use BST property)
    // Time: O(n
), Space: O(H)
    public TreeNode lowestCommonAncestor5(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        
        TreeNode left = lowestCommonAncestor5(root.left, p, q);
        TreeNode right = lowestCommonAncestor5(root.right, p, q);
        
        if (left != null && right != null) {
            return root; // p and q are on different sides
        }
        
        return left != null ? left : right; // Both on same side
    }
}

/*
Algorithm Explanation:

LCA Definition: The lowest common ancestor of nodes p and q is the deepest node 
that has both p and q as descendants (a node can be descendant of itself).

BST Property: For any node in BST:
- All nodes in left subtree have values < node.val
- All nodes in right subtree have values > node.val

Key Insight: In BST, if p.val < root.val < q.val (or vice versa), 
then root is the LCA because p and q are in different subtrees.

Example BST:
        6
       / \
      2   8
     / \ / \
    0  4 7  9
      / \
     3   5

Find LCA of 2 and 8:
- Start at 6: 2 < 6 < 8, so 6 is LCA

Find LCA of 2 and 4:
- Start at 6: both 2,4 < 6, go left
- At 2: 2 ≤ 2 and 4 > 2, so 2 is LCA

Approach 1 (Recursive BST):
1. If both values < root.val: LCA is in left subtree
2. If both values > root.val: LCA is in right subtree  
3. Otherwise: current root is LCA

Approach 2 (Iterative BST):
Same logic as recursive but using iteration.
More space-efficient (O(1) space).

Approach 3 (Path-based):
1. Find path from root to p
2. Find path from root to q
3. LCA is last common node in both paths

Approach 4 (Parent pointers):
1. Collect all ancestors of p
2. Traverse ancestors of q until finding common one

Approach 5 (General binary tree):
Standard LCA algorithm that works for any binary tree.
Doesn't leverage BST property, so less efficient.

Approach Comparison:

1. Recursive BST (Most elegant):
   - Time: O(H), Space: O(H)
   - Clean and intuitive
   - Leverages BST property

2. Iterative BST (Most efficient):
   - Time: O(H), Space: O(1)
   - Best space complexity
   - Preferred for production

3. Path-based:
   - Time: O(H), Space: O(H)
   - Works for general trees
   - More complex implementation

4. Parent pointers:
   - Time: O(H), Space: O(H)
   - Requires modified tree structure
   - Good when parent links available

5. General binary tree:
   - Time: O(n), Space: O(H)
   - Doesn't use BST property
   - Less efficient but more general

Time Complexity Analysis:
- H = height of tree
- Best case (balanced BST): H = log n
- Worst case (skewed BST): H = n
- BST approaches: O(H)
- General approach: O(n)

Space Complexity:
- Recursive: O(H) for call stack
- Iterative: O(1) constant space
- Path-based: O(H) for storing paths

Key Insights:
1. BST property enables efficient navigation
2. LCA is first node where paths to p and q diverge
3. If both nodes on same side, continue searching that side
4. If nodes on different sides, current node is LCA

Edge Cases:
- One node is ancestor of other: ancestor is LCA
- Nodes are same: return either node
- One or both nodes not in tree: handle appropriately
- Root is one of the nodes: root
 is LCA

Optimization Techniques:
1. Use iterative approach for better space complexity
2. Early termination when LCA found
3. Leverage BST property for O(H) instead of O(n)
4. Use parent pointers if available

Common Mistakes:
1. Not leveraging BST property (using general tree algorithm)
2. Incorrect comparison logic (< vs ≤)
3. Not handling case where one node is ancestor of other
4. Forgetting that node can be ancestor of itself
5. Not considering equal values properly

Applications:
- File system directory structures
- Organizational hierarchies
- Phylogenetic trees in biology
- Network routing algorithms
- Version control systems (git merge base)

Follow-up Questions:
1. What if tree is not BST? → Use general binary tree approach
2. What if we need LCA of multiple nodes? → Extend algorithm
3. What if tree is very large? → Consider caching or preprocessing
4. What if queries are frequent? → Preprocess with parent pointers

Variations:
- LCA in general binary tree
- LCA with parent pointers
- LCA of multiple nodes
- Distance between two nodes (using LCA)
*/
```

That completes all 11 Tree problems! Each solution includes:

1. **Multiple approaches** from basic to optimal
2. **Detailed algorithm explanations** with step-by-step examples
3. **Time and space complexity analysis** for each approach
4. **Key insights** and problem-solving strategies
5. **Edge case handling** and common mistakes
6. **Applications** and follow-up questions

The tree problems covered important patterns like:
- **Tree Traversal** (DFS, BFS, Inorder, Preorder, Postorder)
- **Tree Construction** (from traversals)
- **Tree Validation** (BST properties)
- **Tree Properties** (height, paths, structure)
- **Tree Algorithms** (LCA, serialization)

---------------------------------------------------------------


## Linked List Problems Solutions

### 36. Reverse Linked List

```java
/**
 * Problem: Reverse a singly linked list
 * 
 * Multiple approaches: Iterative, Recursive, Stack-based
 */
public class ReverseLinkedList {
    
    // Definition for singly-linked list
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // Approach 1: Iterative - Most efficient and commonly used
    // Time: O(n), Space: O(1)
    public ListNode reverseList1(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode nextTemp = current.next; // Store next node
            current.next = prev;               // Reverse the link
            prev = current;                    // Move prev forward
            current = nextTemp;                // Move current forward
        }
        
        return prev; // prev is now the new head
    }
    
    // Approach 2: Recursive - Elegant but uses O(n) space
    // Time: O(n), Space: O(n)
    public ListNode reverseList2(ListNode head) {
        // Base case: empty list or single node
        if (head == null || head.next == null) {
            return head;
        }
        
        // Recursively reverse the rest of the list
        ListNode newHead = reverseList2(head.next);
        
        // Reverse the current connection
        head.next.next = head;
        head.next = null;
        
        return newHead;
    }
    
    // Approach 3: Using Stack - Intuitive but less efficient
    // Time: O(n), Space: O(n)
    public ListNode reverseList3(ListNode head) {
        if (head == null) return null;
        
        Stack<ListNode> stack = new Stack<>();
        
        // Push all nodes onto stack
        ListNode current = head;
        while (current != null) {
            stack.push(current);
            current = current.next;
        }
        
        // Pop nodes and rebuild list
        ListNode newHead = stack.pop();
        current = newHead;
        
        while (!stack.isEmpty()) {
            current.next = stack.pop();
            current = current.next;
        }
        
        current.next = null; // Important: terminate the list
        return newHead;
    }
    
    // Approach 4: Recursive with helper function
    // Time: O(n), Space: O(n)
    public ListNode reverseList4(ListNode head) {
        return reverseHelper(head, null);
    }
    
    private ListNode reverseHelper(ListNode current, ListNode prev) {
        if (current == null) {
            return prev;
        }
        
        ListNode next = current.next;
        current.next = prev;
        
        return reverseHelper(next, current);
    }
    
    // Approach 5: Two-pass approach (educational)
    // Time: O(
n), Space: O(n)
    public ListNode reverseList5(ListNode head) {
        if (head == null) return null;
        
        // First pass: collect all values
        List<Integer> values = new ArrayList<>();
        ListNode current = head;
        while (current != null) {
            values.add(current.val);
            current = current.next;
        }
        
        // Second pass: create new list in reverse order
        ListNode newHead = new ListNode(values.get(values.size() - 1));
        current = newHead;
        
        for (int i = values.size() - 2; i >= 0; i--) {
            current.next = new ListNode(values.get(i));
            current = current.next;
        }
        
        return newHead;
    }
}

/*
Algorithm Explanation:

Original list: 1 -> 2 -> 3 -> 4 -> 5 -> null
Reversed list: 5 -> 4 -> 3 -> 2 -> 1 -> null

Approach 1 (Iterative):
Key idea: Maintain three pointers (prev, current, next) and reverse links one by one.

Step-by-step trace:
Initial: prev=null, current=1->2->3->4->5->null

Step 1: nextTemp=2->3->4->5->null
        current.next=null (1->null)
        prev=1->null, current=2->3->4->5->null

Step 2: nextTemp=3->4->5->null
        current.next=1->null (2->1->null)
        prev=2->1->null, current=3->4->5->null

Step 3: nextTemp=4->5->null
        current.next=2->1->null (3->2->1->null)
        prev=3->2->1->null, current=4->5->null

Step 4: nextTemp=5->null
        current.next=3->2->1->null (4->3->2->1->null)
        prev=4->3->2->1->null, current=5->null

Step 5: nextTemp=null
        current.next=4->3->2->1->null (5->4->3->2->1->null)
        prev=5->4->3->2->1->null, current=null

Return prev = 5->4->3->2->1->null

Approach 2 (Recursive):
Key idea: Recursively reverse the tail, then fix the current node's connections.

For list 1->2->3->4->5:
1. reverseList(1): calls reverseList(2)
2. reverseList(2): calls reverseList(3)
3. reverseList(3): calls reverseList(4)
4. reverseList(4): calls reverseList(5)
5. reverseList(5): returns 5 (base case)
6. Back to reverseList(4): 
   - newHead = 5
   - 4.next.next = 4 (so 5->4)
   - 4.next = null
   - return 5
7. Continue unwinding...

Approach 3 (Stack):
1. Push all nodes onto stack (LIFO structure naturally reverses)
2. Pop nodes and reconnect them
3. Simple but uses extra space

Approach 4 (Recursive Helper):
Tail recursion version that mimics iterative approach.
Passes previous node as parameter.

Approach 5 (Two-pass):
1. Extract all values into array
2. Create new list from array in reverse order
3. Inefficient but educational

Approach Comparison:

1. Iterative (Best for production):
   - Time: O(n), Space: O(1)
   - Most efficient and preferred
   - No risk of stack overflow

2. Recursive (Elegant):
   - Time: O(n), Space: O(n)
   - Clean and intuitive
   - Risk of stack overflow for large lists

3. Stack-based:
   - Time: O(n), Space: O(n)
   - Easy to understand
   - Extra space overhead

4. Recursive Helper:
   - Time: O(n), Space: O(n)
   - Tail recursion style
   - Similar to iterative logic

5. Two-pass:
   - Time: O(n), Space: O(n)
   - Creates new nodes
   - Not recommended for production

Key Insights:
1. Need to store next pointer before breaking the link
2. Three pointers technique is fundamental
3. Recursive solution works backwards from the end
4. Always handle
 empty list edge case

Visual Representation:
Before: [prev] -> null    1 -> 2 -> 3 -> null
                         [curr]

After:  [prev] -> null <- 1    2 -> 3 -> null
                              [curr]

Edge Cases:
- Empty list (null): return null
- Single node: return same node
- Two nodes: 1->2 becomes 2->1

Common Mistakes:
1. Losing reference to next node before reversing link
2. Not updating all three pointers correctly
3. Forgetting to set last node's next to null
4. Not handling empty list case
5. Stack overflow with recursive approach on large lists

Optimization Notes:
- Iterative approach is optimal for space and time
- Recursive approach is good for understanding
- Always prefer iterative for production code
- Consider tail recursion optimization if available

Applications:
- Undo operations in text editors
- Browser back button functionality
- Reversing data streams
- Algorithm building blocks (merge sort, etc.)
- Stack implementation using linked list

Follow-up Questions:
1. Reverse only part of the list (between positions m and n)
2. Reverse in groups of k nodes
3. Reverse alternate nodes
4. Check if list is palindrome (using reversal)

Memory Management:
- Iterative: O(1) extra space
- Recursive: O(n) call stack space
- Stack: O(n) for explicit stack
- All approaches modify original list in-place (except two-pass)

Testing Strategy:
- Empty list: null -> null
- Single node: 1 -> 1
- Two nodes: 1->2 -> 2->1
- Multiple nodes: 1->2->3 -> 3->2->1
- Large list: test for stack overflow with recursive
*/
```

### 37. Detect Cycle in Linked List

```java
import java.util.*;

/**
 * Problem: Detect if linked list has a cycle
 * 
 * Multiple approaches: Floyd's Algorithm, HashSet, Marking
 */
public class LinkedListCycle {
    
    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
            next = null;
        }
    }
    
    // Approach 1: Floyd's Cycle Detection (Tortoise and Hare) - Optimal
    // Time: O(n), Space: O(1)
    public boolean hasCycle1(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        
        ListNode slow = head;      // Tortoise: moves 1 step
        ListNode fast = head.next; // Hare: moves 2 steps
        
        while (slow != fast) {
            // If fast reaches end, no cycle
            if (fast == null || fast.next == null) {
                return false;
            }
            
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return true; // Slow and fast met, cycle detected
    }
    
    // Approach 2: HashSet to track visited nodes
    // Time: O(n), Space: O(n)
    public boolean hasCycle2(ListNode head) {
        Set<ListNode> visited = new HashSet<>();
        
        ListNode current = head;
        while (current != null) {
            if (visited.contains(current)) {
                return true; // Found a node we've seen before
            }
            visited.add(current);
            current = current.next;
        }
        
        return false; // Reached end without finding cycle
    }
    
    // Approach 3: Marking nodes (modifies original list)
    // Time: O(n), Space: O(1)
    public boolean hasCycle3(ListNode head) {
        ListNode current = head;
        
        while (current != null) {
            // If we've seen this node before (marked with special
 value)
            if (current.val == Integer.MIN_VALUE) {
                return true;
            }
            
            // Mark current node as visited
            current.val = Integer.MIN_VALUE;
            current = current.next;
        }
        
        return false;
    }
    
    // Approach 4: Limit-based detection (not reliable but educational)
    // Time: O(n), Space: O(1)
    public boolean hasCycle4(ListNode head) {
        int limit = 10000; // Assume list won't be longer than this
        ListNode current = head;
        
        for (int i = 0; i < limit && current != null; i++) {
            current = current.next;
        }
        
        // If we haven't reached end after limit steps, assume cycle
        return current != null;
    }
    
    // Approach 5: Floyd's with same starting position
    // Time: O(n), Space: O(1)
    public boolean hasCycle5(ListNode head) {
        if (head == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        // Move pointers until they meet or fast reaches end
        do {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        } while (slow != fast);
        
        return true;
    }
    
    // Bonus: Find the start of the cycle (Floyd's extended algorithm)
    // Time: O(n), Space: O(1)
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        
        // Phase 1: Detect if cycle exists
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) {
                break; // Cycle detected
            }
        }
        
        // No cycle found
        if (fast == null || fast.next == null) {
            return null;
        }
        
        // Phase 2: Find start of cycle
        // Move one pointer to head, keep other at meeting point
        // Move both at same speed until they meet
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        
        return slow; // Start of cycle
    }
}

/*
Algorithm Explanation:

A cycle exists if a node's next pointer points to a previously visited node,
creating a loop in the linked list.

Example with cycle:
1 -> 2 -> 3 -> 4
     ^         |
     |_________|

Example without cycle:
1 -> 2 -> 3 -> 4 -> null

Approach 1 (Floyd's Cycle Detection):
Use two pointers moving at different speeds:
- Slow pointer: moves 1 step at a time
- Fast pointer: moves 2 steps at a time

If there's a cycle, fast pointer will eventually catch up to slow pointer.
If no cycle, fast pointer will reach the end.

Mathematical proof:
- Let's say cycle length is C
- When slow enters cycle, fast is already inside
- Fast gains 1 position on slow each iteration
- They will meet within C iterations

Step-by-step trace for cyclic list:
1 -> 2 -> 3 -> 4 -> 2 (cycle back to 2)

Initial: slow=1, fast=2
Step 1:  slow=2, fast=4
Step 2:  slow=3, fast=3 (met! cycle detected)

Approach 2 (HashSet):
Store every visited node in a set.
If we encounter a node already in the set, there's a cycle.

Approach 3 (Marking):
Modify node values to mark them as visited.
If we see a marked node again, there's a cycle.
Note: This destroys original data.

Approach 4 (Limit-based):
Assume if we traverse more than a
 certain number of nodes,
there must be a cycle. Not reliable for general use.

Approach 5 (Floyd's variant):
Same as Approach 1 but both pointers start at head.
Use do-while loop to handle the initial equality.

Approach Comparison:

1. Floyd's Algorithm (Best):
   - Time: O(n), Space: O(1)
   - No extra space needed
   - Doesn't modify original list
   - Industry standard

2. HashSet:
   - Time: O(n), Space: O(n)
   - Easy to understand
   - Uses extra memory
   - Good for debugging

3. Marking:
   - Time: O(n), Space: O(1)
   - Destroys original data
   - Not practical for most cases
   - Only works if values can be modified

4. Limit-based:
   - Time: O(n), Space: O(1)
   - Unreliable and not recommended
   - Educational purpose only

5. Floyd's variant:
   - Time: O(n), Space: O(1)
   - Same efficiency as approach 1
   - Slightly different implementation

Extended Algorithm (Cycle Start Detection):

Phase 1: Detect cycle using Floyd's algorithm
Phase 2: Find start of cycle

Mathematical insight:
- Let distance from head to cycle start = a
- Let distance from cycle start to meeting point = b
- Let cycle length = c

When pointers meet:
- Slow traveled: a + b
- Fast traveled: a + b + c (one extra cycle)
- Since fast travels twice as fast: 2(a + b) = a + b + c
- Solving: a = c - b

This means: distance from head to cycle start = 
           distance from meeting point to cycle start

So if we move one pointer to head and keep other at meeting point,
moving both at same speed, they'll meet at cycle start.

Time Complexity Analysis:
- Floyd's: O(n) - each node visited at most twice
- HashSet: O(n) - each node visited once
- Marking: O(n) - each node visited once

Space Complexity:
- Floyd's: O(1) - only two pointers
- HashSet: O(n) - store up to n nodes
- Marking: O(1) - no extra space

Key Insights:
1. Floyd's algorithm is optimal for cycle detection
2. Two pointers at different speeds will meet if cycle exists
3. Mathematical relationship enables finding cycle start
4. Constant space solution is always preferred

Edge Cases:
- Empty list: no cycle
- Single node pointing to itself: cycle
- Single node pointing to null: no cycle
- Two nodes forming cycle: cycle
- Long list with cycle at end: cycle

Common Mistakes:
1. Not checking for null pointers in fast pointer movement
2. Starting both pointers at same position without proper loop
3. Not handling empty list or single node cases
4. Infinite loop when cycle exists (in naive approaches)
5. Off-by-one errors in pointer movements

Applications:
- Memory leak detection
- Infinite loop detection in algorithms
- Graph cycle detection (using similar principles)
- Deadlock detection in operating systems
- Validation of data structures

Follow-up Questions:
1. Find the start of the cycle
2. Find the length of the cycle
3. Remove the cycle from the list
4. Detect cycle in a graph
5. Find if two linke
d lists intersect

Optimization Notes:
- Floyd's algorithm is already optimal
- Can optimize constant factors but not asymptotic complexity
- Early termination when fast reaches null
- Consider using fast.next.next directly to avoid extra null checks

Testing Strategy:
- No cycle: 1->2->3->null
- Self loop: 1->1
- Two node cycle: 1->2->1
- Cycle at end: 1->2->3->2
- Cycle at beginning: 1->2->1->2->...
- Large cycle: test performance
*/
```

### 38. Merge Two Sorted Lists

```java
/**
 * Problem: Merge two sorted linked lists into one sorted list
 * 
 * Multiple approaches: Iterative, Recursive, In-place
 */
public class MergeTwoSortedLists {
    
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // Approach 1: Iterative with dummy head - Most efficient
    // Time: O(m + n), Space: O(1)
    public ListNode mergeTwoLists1(ListNode list1, ListNode list2) {
        // Create dummy head to simplify edge cases
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        // Merge while both lists have nodes
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }
        
        // Append remaining nodes (at most one list has remaining nodes)
        current.next = (list1 != null) ? list1 : list2;
        
        return dummy.next; // Return head of merged list
    }
    
    // Approach 2: Recursive - Elegant but uses O(m + n) space
    // Time: O(m + n), Space: O(m + n)
    public ListNode mergeTwoLists2(ListNode list1, ListNode list2) {
        // Base cases
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        
        // Choose smaller head and recursively merge rest
        if (list1.val <= list2.val) {
            list1.next = mergeTwoLists2(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists2(list1, list2.next);
            return list2;
        }
    }
    
    // Approach 3: Iterative without dummy head
    // Time: O(m + n), Space: O(1)
    public ListNode mergeTwoLists3(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        
        // Determine head of merged list
        ListNode head, current;
        if (list1.val <= list2.val) {
            head = current = list1;
            list1 = list1.next;
        } else {
            head = current = list2;
            list2 = list2.next;
        }
        
        // Merge remaining nodes
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }
        
        // Append remaining nodes
        current.next = (list1 != null) ? list1 : list2;
        
        return head;
    }
    
    // Approach 4: Using priority queue (overkill but educational)
    // Time: O((m + n) log(m + n)), Space: O(m + n)
    public ListNode mergeTwoLists4(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        
        // Priority queue to maintain sorted order
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        // Add all nodes to priority queue
        while (list1 != null) {
            pq.offer(list1);
            list1 = list1.next;
        }
        while (list2 != null) {
            pq.offer(list2);
            list2 = list2.next;
        }
        
        // Build result list from priority queue
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!pq.isEmpty()) {
            current.next = pq.poll();
            current = current.next;
        }
        
        current.next = null; // Important: terminate the list
        return dummy.next;
    }
    
    // Approach 5: In-place merge (modifies input lists)
    // Time: O(m + n), Space: O(1)
    public ListNode mergeTwoLists5(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        
        // Ensure list1 starts
 with smaller value
        if (list1.val > list2.val) {
            ListNode temp = list1;
            list1 = list2;
            list2 = temp;
        }
        
        ListNode current = list1;
        
        while (current.next != null && list2 != null) {
            if (current.next.val <= list2.val) {
                current = current.next;
            } else {
                // Insert list2 node between current and current.next
                ListNode temp = list2.next;
                list2.next = current.next;
                current.next = list2;
                current = list2;
                list2 = temp;
            }
        }
        
        // Append remaining nodes from list2
        if (list2 != null) {
            current.next = list2;
        }
        
        return list1;
    }
}

/*
Algorithm Explanation:

Merge two sorted lists by comparing heads and choosing smaller one.
Continue until one list is exhausted, then append the other.

Example:
list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4

Step-by-step merge (Approach 1):
1. dummy -> null, current = dummy
2. Compare 1 and 1: choose list1, current.next = 1
   dummy -> 1, current = 1, list1 = 2->4
3. Compare 2 and 1: choose list2, current.next = 1
   dummy -> 1 -> 1, current = 1, list2 = 3->4
4. Compare 2 and 3: choose list1, current.next = 2
   dummy -> 1 -> 1 -> 2, current = 2, list1 = 4
5. Compare 4 and 3: choose list2, current.next = 3
   dummy -> 1 -> 1 -> 2 -> 3, current = 3, list2 = 4
6. Compare 4 and 4: choose list1, current.next = 4
   dummy -> 1 -> 1 -> 2 -> 3 -> 4, current = 4, list1 = null
7. list1 is null, append list2: current.next = 4
   Final: 1 -> 1 -> 2 -> 3 -> 4 -> 4

Approach 1 (Iterative with Dummy):
- Use dummy head to avoid special cases for empty result
- Compare heads of both lists and choose smaller
- Advance pointer in chosen list
- Continue until one list is empty
- Append remaining list

Approach 2 (Recursive):
- Base cases: if one list is empty, return the other
- Choose smaller head and recursively merge rest
- Clean and elegant but uses call stack space

Approach 3 (Iterative without Dummy):
- Handle head selection separately
- Same logic as approach 1 but more complex head handling
- Slightly more efficient (no dummy node)

Approach 4 (Priority Queue):
- Add all nodes to priority queue
- Extract nodes in sorted order
- Overkill for two lists but generalizes to k lists

Approach 5 (In-place):
- Modify one of the input lists to create result
- More complex pointer manipulation
- Saves space by reusing existing nodes

Approach Comparison:

1. Iterative with Dummy (Best for interviews):
   - Time: O(m + n), Space: O(1)
   - Clean and easy to understand
   - Handles edge cases elegantly

2. Recursive (Most elegant):
   - Time: O(m + n), Space: O(m + n)
   - Very clean code
   - Risk of stack overflow for large lists

3. Iterative without Dummy:
   - Time: O(m + n), Space: O(1)
   - Slightly more efficient
   - More complex edge case handling

4. Priority Queue:
   - Time: O((m + n) log(m + n)), Space: O(m + n)
   - Overkill for two lists
   - Good foundation for k-way merge

5. In-place:
   - Time: O(m + n), Space: O(1)
   - Most space efficient
   - Complex implementation

Key Insights:
1. Dummy head simpl
ifies edge case handling
2. Only need to compare current heads of both lists
3. When one list is exhausted, append the other entirely
4. Maintain sorted order by always choosing smaller element

Edge Cases:
- Both lists empty: return null
- One list empty: return the other
- Lists of different lengths: append remaining
- All elements in one list smaller: simple append
- Duplicate values: maintain stability (keep relative order)

Common Mistakes:
1. Not handling empty lists properly
2. Forgetting to append remaining nodes
3. Not advancing pointers correctly
4. Creating cycles in the result list
5. Not terminating the result list properly

Optimization Techniques:
1. Use dummy head to simplify code
2. Early termination when one list is empty
3. Avoid unnecessary comparisons
4. Reuse existing nodes instead of creating new ones

Applications:
- Merge sort implementation
- Database join operations
- Merging sorted files
- Combining sorted streams
- Priority queue operations

Follow-up Questions:
1. Merge k sorted lists
2. Merge in descending order
3. Merge with custom comparator
4. Merge and remove duplicates
5. Merge with size constraints

Memory Considerations:
- Approaches 1, 3, 5: O(1) extra space
- Approach 2: O(m + n) call stack space
- Approach 4: O(m + n) for priority queue
- All approaches reuse existing nodes (no new node creation)

Testing Strategy:
- Empty lists: [], [] -> []
- One empty: [1,2], [] -> [1,2]
- Same length: [1,3], [2,4] -> [1,2,3,4]
- Different lengths: [1], [2,3,4] -> [1,2,3,4]
- Duplicates: [1,1], [1,2] -> [1,1,1,2]
- All elements in one list smaller: [1,2], [3,4] -> [1,2,3,4]

Performance Notes:
- Linear time complexity is optimal (must examine all elements)
- Constant space is achievable and preferred
- Iterative approach avoids recursion overhead
- Dummy head technique is widely applicable
*/
```

### 39. Merge k Sorted Lists

```java
import java.util.*;

/**
 * Problem: Merge k sorted linked lists into one sorted list
 * 
 * Multiple approaches: Divide & Conquer, Priority Queue, Sequential Merge
 */
public class MergeKSortedLists {
    
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // Approach 1: Divide and Conquer - Most efficient
    // Time: O(N log k) where N is total nodes, k is number of lists
    // Space: O(log k) for recursion stack
    public ListNode mergeKLists1(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        return mergeKListsHelper(lists, 0, lists.length - 1);
    }
    
    private ListNode mergeKListsHelper(ListNode[] lists, int start, int end) {
        if (start == end) {
            return lists[start];
        }
        
        if (start + 1 == end) {
            return mergeTwoLists(lists[start], lists[end]);
        }
        
        int mid = start + (end - start) / 2;
        ListNode left = mergeKListsHelper(lists, start, mid);
        ListNode right = mergeKListsHelper(lists, mid + 1, end);
        
        return mergeTwoLists(left, right);
    }
    
    // Helper method to merge two sorted lists
    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }
        
        current.next = (l1 != null) ? l1 : l2;
        return dummy.next;
    }
    
    // Approach 2: Priority Queue (Min Heap) - Intuitive
    // Time: O(N log k), Space: O(k)
    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length ==
 0) {
            return null;
        }
        
        // Priority queue to maintain k smallest elements
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        // Add head of each non-empty list to priority queue
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!pq.isEmpty()) {
            // Get the smallest element
            ListNode smallest = pq.poll();
            current.next = smallest;
            current = current.next;
            
            // Add next element from the same list
            if (smallest.next != null) {
                pq.offer(smallest.next);
            }
        }
        
        return dummy.next;
    }
    
    // Approach 3: Sequential merge - Simple but less efficient
    // Time: O(N * k), Space: O(1)
    public ListNode mergeKLists3(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        ListNode result = null;
        
        for (ListNode list : lists) {
            result = mergeTwoLists(result, list);
        }
        
        return result;
    }
    
    // Approach 4: Iterative divide and conquer - Space optimized
    // Time: O(N log k), Space: O(1)
    public ListNode mergeKLists4(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        int interval = 1;
        
        while (interval < lists.length) {
            for (int i = 0; i + interval < lists.length; i += interval * 2) {
                lists[i] = mergeTwoLists(lists[i], lists[i + interval]);
            }
            interval *= 2;
        }
        
        return lists[1];
    }
    
    // Approach 5: Convert to array, sort, and rebuild - Different perspective
    // Time: O(N log N), Space: O(N)
    public ListNode mergeKLists5(ListNode[] lists) {
        List<Integer> values = new ArrayList<>();
        
        // Collect all values
        for (ListNode list : lists) {
            ListNode current = list;
            while (current != null) {
                values.add(current.val);
                current = current.next;
            }
        }
        
        // Sort values
        Collections.sort(values);
        
        // Build result list
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        for (int val : values) {
            current.next = new ListNode(val);
            current = current.next;
        }
        
        return dummy.next;
    }
    
    // Approach 6: Using merge sort concept with lists
    // Time: O(N log k), Space: O(log k)
    public ListNode mergeKLists6(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        
        List<ListNode> listCollection = new ArrayList<>();
        for (ListNode list : lists) {
            if (list != null) {
                listCollection.add(list);
            }
        }
        
        while (listCollection.size() > 1) {
            List<ListNode> mergedLists = new ArrayList<>();
            
            for (int i = 0; i < listCollection.size(); i += 2) {
                ListNode l1 = listCollection.get(i);
                ListNode l2 = (i + 1 < listCollection.size()) ? listCollection.get(i + 1) : null;
                mergedLists.add(mergeTwoLists(l1, l2));
            }
            
            listCollection = mergedLists;
        }
        
        return listCollection.get(0);
    }
}

/*
Algorithm Explanation:

Problem: Given k sorted linked lists, merge them into one sorted list.

Example:
Input: [
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6

Approach 1 (Divide and Conquer):
Recursively divide the k lists into two halves and merge them.
Similar to merge sort but for linked lists.

Tree structure for k=4 lists:
       merge(0,3)
      /          \
  merge(0,1)    merge(2,3)
   /    \        /    \
list0 list1   list2 list3

Time complexity: O(N log k)
- log k levels in recursion tree
- Each level processes all N nodes
- Each merge operation is O(length of lists being merged)

Approach 2 (Priority Queue):
Maintain a min-heap of size k containing the current
 smallest element from each list.
Extract minimum, add to result, and insert next element from same list.

Step-by-step for example:
1. PQ: [1(list0), 1(list1), 2(list2)]
2. Extract 1(list0), add 4: PQ: [1(list1), 2(list2), 4(list0)]
3. Extract 1(list1), add 3: PQ: [2(list2), 3(list1), 4(list0)]
4. Extract 2(list2), add 6: PQ: [3(list1), 4(list0), 6(list2)]
5. Continue...

Approach 3 (Sequential Merge):
Merge lists one by one: result = merge(result, list[i])
Simple but inefficient for large k.

Time analysis:
- Merge 1st and 2nd: O(n1 + n2)
- Merge result with 3rd: O(n1 + n2 + n3)
- ...
- Total: O(N * k) where N is total nodes

Approach 4 (Iterative Divide and Conquer):
Bottom-up approach, merge pairs iteratively.
Space-optimized version of approach 1.

Iteration pattern:
Round 1: merge(0,1), merge(2,3), merge(4,5), ...
Round 2: merge(0,2), merge(4,6), ...
Round 3: merge(0,4), ...

Approach 5 (Sort Array):
Extract all values, sort, and rebuild list.
Different approach but less efficient.

Approach 6 (List-based Merge Sort):
Similar to approach 1 but using list operations.
More intuitive for some developers.

Approach Comparison:

1. Divide and Conquer (Best overall):
   - Time: O(N log k), Space: O(log k)
   - Optimal time complexity
   - Recursive implementation

2. Priority Queue (Most intuitive):
   - Time: O(N log k), Space: O(k)
   - Easy to understand and implement
   - Good for streaming scenarios

3. Sequential Merge (Simplest):
   - Time: O(N * k), Space: O(1)
   - Simple but inefficient for large k
   - Good for small k

4. Iterative Divide and Conquer:
   - Time: O(N log k), Space: O(1)
   - Space-optimized version of approach 1
   - No recursion overhead

5. Sort Array:
   - Time: O(N log N), Space: O(N)
   - Creates new nodes
   - Not optimal but educational

6. List-based Merge Sort:
   - Time: O(N log k), Space: O(log k)
   - Alternative implementation
   - More memory for intermediate lists

Key Insights:
1. Divide and conquer achieves optimal O(N log k) time
2. Priority queue provides intuitive O(N log k) solution
3. Sequential merge is O(N * k) - avoid for large k
4. Space can be optimized to O(1) with iterative approach

Time Complexity Analysis:
- N = total number of nodes across all lists
- k = number of lists
- Optimal: O(N log k)
- Naive: O(N * k)

Space Complexity:
- Recursive: O(log k) for call stack
- Priority Queue: O(k) for heap
- Iterative: O(1) constant space

Edge Cases:
- Empty array: return null
- Array with nulls: filter out nulls
- Single list: return that list
- All empty lists: return null
- Lists of very different sizes: algorithm handles naturally

Common Mistakes:
1. Not handling null lists in input array
2. Incorrect priority queue comparator
3. Forgetting to add next element to priority queue
4. Off-by-one errors in divide and conquer
5. Not terminating result list properly

Optimization Techniques:
1. Filter out null lists before processing
2. Use iterative approach to save stack space
3. Early termination when priority
 queue is empty
4. Reuse existing nodes instead of creating new ones

Applications:
- Database merge operations
- External sorting
- Merging sorted files
- Distributed system data aggregation
- Stream processing

Follow-up Questions:
1. What if lists are not sorted?
2. How to handle very large k (memory constraints)?
3. Merge with custom comparator
4. Remove duplicates while merging
5. Find kth smallest element across all lists

Performance Considerations:
- For small k: sequential merge might be faster due to simplicity
- For large k: divide and conquer or priority queue
- Memory constraints: prefer iterative approaches
- Real-time processing: priority queue for streaming
*/
```

### 40. Remove Nth Node From End of List

```java
/**
 * Problem: Remove the nth node from the end of linked list
 * 
 * Multiple approaches: Two-pass, One-pass with two pointers, Stack-based
 */
public class RemoveNthNodeFromEnd {
    
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // Approach 1: Two-pass solution - Simple and intuitive
    // Time: O(L), Space: O(1) where L is length of list
    public ListNode removeNthFromEnd1(ListNode head, int n) {
        // First pass: calculate length
        int length = 0;
        ListNode current = head;
        while (current != null) {
            length++;
            current = current.next;
        }
        
        // Handle edge case: remove head
        if (n == length) {
            return head.next;
        }
        
        // Second pass: find node before the one to remove
        current = head;
        for (int i = 0; i < length - n - 1; i++) {
            current = current.next;
        }
        
        // Remove the nth node from end
        current.next = current.next.next;
        
        return head;
    }
    
    // Approach 2: One-pass with two pointers - Most efficient
    // Time: O(L), Space: O(1)
    public ListNode removeNthFromEnd2(ListNode head, int n) {
        // Use dummy head to handle edge cases
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        
        ListNode first = dummy;
        ListNode second = dummy;
        
        // Move first pointer n+1 steps ahead
        for (int i = 0; i <= n; i++) {
            first = first.next;
        }
        
        // Move both pointers until first reaches end
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        
        // Remove the nth node from end
        second.next = second.next.next;
        
        return dummy.next;
    }
    
    // Approach 3: Using stack - Alternative approach
    // Time: O(L), Space: O(L)
    public ListNode removeNthFromEnd3(ListNode head, int n) {
        Stack<ListNode> stack = new Stack<>();
        
        // Push all nodes onto stack
        ListNode current = head;
        while (current != null) {
            stack.push(current);
            current = current.next;
        }
        
        // Pop n nodes to reach the node to remove
        for (int i = 0; i < n; i++) {
            stack.pop();
        }
        
        // Handle edge case: remove head
        if (stack.isEmpty()) {
            return head.next;
        }
        
        // Remove the nth node from end
        ListNode prev = stack.peek();
        prev.next = prev.next.next;
        
        return head;
    }
    
    // Approach 4: Recursive approach - Elegant but uses call stack
    // Time: O(L), Space: O(L)
    public ListNode removeNthFromEnd4(ListNode head, int n) {
        int[] count = new int[2]; // Use array to pass by reference
        return removeNthHelper(head, n, count);
    }
    
    private ListNode removeNthHelper(ListNode node, int n, int[] count) {
        if (node == null) {
            return null;
        }
        
        node.next = removeNthHelper(node.next, n, count);
        count[1]++;
        
        // If this is the nth node from end, remove it
        if (count[1] == n) {
            return node.next;
        }
        
        return node;
    }
    
    // Approach 5: Convert to array and rebuild - Educational
    // Time: O(L), Space: O(L)
    public ListNode removeNthFromEnd5(ListNode head, int n) {
        // Convert to array
        List<ListNode> nodes = new ArrayList<>();
        ListNode current = head;
        while (current != null) {
            nodes.add(current);
            current = current.next;
        }
        
        int length = nodes.size();
        
        // Handle
 edge case: remove head
        if (n == length) {
            return head.next;
        }
        
        // Remove nth node from end
        int indexToRemove = length - n;
        if (indexToRemove > 0) {
            nodes.get(indexToRemove - 1).next = nodes.get(indexToRemove).next;
        }
        
        return head;
    }
    
    // Approach 6: Two pointers with explicit gap - Clearer logic
    // Time: O(L), Space: O(1)
    public ListNode removeNthFromEnd6(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        
        ListNode slow = dummy;
        ListNode fast = dummy;
        
        // Create gap of n+1 between slow and fast
        for (int i = 0; i <= n; i++) {
            if (fast == null) {
                // n is larger than list length
                return head;
            }
            fast = fast.next;
        }
        
        // Move both pointers until fast reaches end
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        
        // slow.next is the node to remove
        slow.next = slow.next.next;
        
        return dummy.next;
    }
}

/*
Algorithm Explanation:

Problem: Remove the nth node from the end of a linked list.
Note: n is 1-indexed (1st from end is the last node).

Example:
Input: 1->2->3->4->5, n=2
Output: 1->2->3->5 (removed 4, which is 2nd from end)

Approach 1 (Two-pass):
1. First pass: count total length L
2. Second pass: go to (L-n)th node from beginning
3. Remove the next node

For example above:
- Length = 5
- Node to remove is at position 5-2 = 3 (0-indexed)
- Go to position 2, remove position 3

Approach 2 (One-pass with two pointers):
Use two pointers with a gap of n+1 between them.
When fast pointer reaches end, slow pointer is at the node before the one to remove.

Step-by-step for example:
1. dummy->1->2->3->4->5, n=2
2. Move fast n+1=3 steps: fast at 3, slow at dummy
3. Move both until fast reaches end:
   - fast at 4, slow at 1
   - fast at 5, slow at 2  
   - fast at null, slow at 3
4. slow.next (4) is the node to remove
5. Set slow.next = slow.next.next

Approach 3 (Stack):
1. Push all nodes onto stack
2. Pop n nodes to reach the node to remove
3. The top of stack is the previous node
4. Remove the target node

Approach 4 (Recursive):
1. Recursively traverse to end
2. Count nodes on the way back
3. When count equals n, remove that node

Approach 5 (Array conversion):
1. Convert linked list to array
2. Calculate index to remove
3. Update pointers accordingly

Approach 6 (Explicit gap):
Similar to approach 2 but with clearer gap creation logic.

Approach Comparison:

1. Two-pass (Simple):
   - Time: O(L), Space: O(1)
   - Easy to understand
   - Requires two traversals

2. One-pass Two Pointers (Optimal):
   - Time: O(L), Space: O(1)
   - Single traversal
   - Most efficient approach

3. Stack-based:
   - Time: O(L), Space: O(L)
   - Intuitive but uses extra space
   - Good for understanding

4. Recursive:
   - Time: O(L), Space: O(L)
   - Elegant but uses call stack
   - Risk of stack overflow

5. Array conversion:
   - Time: O(L), Space: O(L)
   - Educational but inefficient
   - Creates extra data structure

6. Explicit gap:
   - Time: O(L), Space: O(1)
   - Clearer version of approach 2
   - Good for interviews

Key Insights:
1. Dummy head simplifies edge case
 handling
2. Two pointers with fixed gap is elegant solution
3. nth from end = (length - n)th from beginning
4. Always handle edge case where head is removed

Edge Cases:
- Remove head (n equals list length): return head.next
- Single node list (n=1): return null
- n larger than list length: return original list
- Empty list: return null

Common Mistakes:
1. Not handling removal of head node
2. Off-by-one errors in gap calculation
3. Not using dummy head for edge cases
4. Incorrect pointer advancement
5. Not checking for null pointers

Optimization Techniques:
1. Use dummy head to avoid special cases
2. Single pass with two pointers
3. Early validation of n vs list length
4. Reuse existing nodes instead of creating new ones

Visual Representation (Two Pointers):
Initial: dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null, n=2
         slow    fast (gap of n+1=3)

Step 1:  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
         slow              fast

Step 2:  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                   slow         fast

Step 3:  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                          slow         fast

Step 4:  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                                 slow    fast

Now slow.next (4) is the node to remove.

Applications:
- Undo operations (remove last n operations)
- Buffer management (remove old entries)
- Sliding window algorithms
- List manipulation in general

Follow-up Questions:
1. Remove nth node from beginning
2. Remove all nodes from position m to n
3. Remove nodes with specific values
4. Remove duplicates while maintaining order

Performance Analysis:
- Best approach: One-pass with two pointers
- Time: O(L) - must traverse list at least once
- Space: O(1) - only need a few pointers
- Single traversal is optimal

Testing Strategy:
- Normal case: [1,2,3,4,5], n=2 -> [1,2,3,5]
- Remove head: [1,2], n=2 -> [2]
- Single node: [1], n=1 -> []
- Remove tail: [1,2,3], n=1 -> [1,2]
- Large n: [1,2], n=3 -> [1,2] (no change)
*/
```

### 41. Reorder List

```java
import java.util.*;

/**
 * Problem: Reorder list L0→L1→…→Ln-1→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→…
 * 
 * Multiple approaches: Stack, Array, Two pointers with reversal
 */
public class ReorderList {
    
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // Approach 1: Find middle + Reverse + Merge - Most efficient
    // Time: O(n), Space: O(1)
    public void reorderList1(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        // Step 1: Find the middle of the list
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // Step 2: Reverse the second half
        ListNode secondHalf = reverseList(slow.next);
        slow.next = null; // Cut the list into two hal
ves
        
        // Step 3: Merge the two halves alternately
        ListNode first = head;
        ListNode second = secondHalf;
        
        while (second != null) {
            ListNode temp1 = first.next;
            ListNode temp2 = second.next;
            
            first.next = second;
            second.next = temp1;
            
            first = temp1;
            second = temp2;
        }
    }
    
    // Helper method to reverse a linked list
    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
    
    // Approach 2: Using Stack - Intuitive but uses extra space
    // Time: O(n), Space: O(n)
    public void reorderList2(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        // Push all nodes onto stack
        Stack<ListNode> stack = new Stack<>();
        ListNode current = head;
        while (current != null) {
            stack.push(current);
            current = current.next;
        }
        
        // Reorder by alternating between start and end
        current = head;
        int count = 0;
        int totalNodes = stack.size();
        
        while (count < totalNodes / 2) {
            ListNode next = current.next;
            ListNode last = stack.pop();
            
            current.next = last;
            last.next = next;
            current = next;
            count++;
        }
        
        current.next = null; // Important: terminate the list
    }
    
    // Approach 3: Convert to array and rebuild - Simple but inefficient
    // Time: O(n), Space: O(n)
    public void reorderList3(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        // Convert to array
        List<ListNode> nodes = new ArrayList<>();
        ListNode current = head;
        while (current != null) {
            nodes.add(current);
            current = current.next;
        }
        
        // Rebuild with reordered connections
        int left = 0;
        int right = nodes.size() - 1;
        
        while (left < right) {
            nodes.get(left).next = nodes.get(right);
            left++;
            
            if (left < right) {
                nodes.get(right).next = nodes.get(left);
                right--;
            }
        }
        
        nodes.get(left).next = null; // Terminate the list
    }
    
    // Approach 4: Recursive approach - Elegant but uses call stack
    // Time: O(n), Space: O(n)
    public void reorderList4(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        int length = getLength(head);
        reorderHelper(head, length);
    }
    
    private ListNode reorderHelper(ListNode head, int length) {
        if (length == 1) {
            ListNode tail = head.next;
            head.next = null;
            return tail;
        }
        
        if (length == 2) {
            ListNode tail = head.next.next;
            head.next.next = null;
            return tail;
        }
        
        ListNode tail = reorderHelper(head.next, length - 2);
        ListNode nextTail = tail.next;
        tail.next = head.next;
        head.next = tail;
        
        return nextTail;
    }
    
    private int getLength(ListNode head) {
        int length = 0;
        while (head != null) {
            length++;
            head = head.next;
        }
        return length;
    }
    
    // Approach 5: Two pointers with deque simulation
    // Time: O(n), Space: O(n)
    public void reorderList5(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        // Collect all nodes in deque
        Deque<ListNode> deque = new ArrayDeque<>();
        ListNode current = head;
        while (current != null) {
            deque.addLast(current);
            current = current.next;
        }
        
        // Remove head from deque since it's already positioned
        deque.removeFirst();
        
        current = head;
        boolean takeFromEnd = true;
        
        while (!deque.isEmpty()) {
            if (takeFromEnd) {
                current.next = deque.removeLast();
            } else {
                current.next = deque.removeFirst();
            }
            current = current.next;
            takeFromEnd = !takeFromEnd;
        }
        
        current.next = null; // Terminate the list
    }
    
    // Approach 6: Iterative with length calculation
    // Time: O(n), Space: O(1)
    public void reorderList6(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        
        // Calculate length
        int length = 0;
        ListNode current = head;
        while (current != null) {
            length++;
            current = current.next;
        }
        
        // Find middle
        current = head;
        for (int i = 0; i < (length - 1) / 2; i++) {
            current = current.next;
        }
        
        // Reverse second half
        ListNode secondHalf = reverseList(current.next);
        current.next = null;
        
        // Merge altern
ately
        ListNode first = head;
        ListNode second = secondHalf;
        
        while (second != null) {
            ListNode temp1 = first.next;
            ListNode temp2 = second.next;
            
            first.next = second;
            second.next = temp1;
            
            first = temp1;
            second = temp2;
        }
    }
}

/*
Algorithm Explanation:

Problem: Reorder linked list from L0→L1→…→Ln-1→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→…

Example:
Input:  1->2->3->4->5
Output: 1->5->2->4->3

Input:  1->2->3->4
Output: 1->4->2->3

Approach 1 (Find Middle + Reverse + Merge):
This is the optimal solution with three main steps:

Step 1: Find the middle of the list
- Use slow/fast pointers (Floyd's algorithm)
- Slow moves 1 step, fast moves 2 steps
- When fast reaches end, slow is at middle

Step 2: Reverse the second half
- Cut the list at middle
- Reverse the second half using standard reversal

Step 3: Merge alternately
- Take nodes alternately from first and second half
- Connect them in the required pattern

Detailed trace for [1,2,3,4,5]:

Step 1 - Find middle:
slow=1, fast=1: fast.next=2, fast.next.next=3, continue
slow=2, fast=3: fast.next=4, fast.next.next=5, continue  
slow=3, fast=5: fast.next=null, stop
Middle found at slow=3

Step 2 - Reverse second half:
Original: 1->2->3->4->5
Cut: 1->2->3, 4->5
Reverse second: 1->2->3, 5->4

Step 3 - Merge:
first=1->2->3, second=5->4
Merge: 1->5->2->4->3

Approach 2 (Stack):
1. Push all nodes onto stack
2. Alternate between taking from beginning and popping from stack
3. Stack naturally gives us nodes from the end

Approach 3 (Array):
1. Convert linked list to array for random access
2. Use two pointers (left, right) to access from both ends
3. Rebuild connections in required order

Approach 4 (Recursive):
1. Recursively process the middle portion
2. Connect current node with tail node
3. Return the new tail for parent call

Approach 5 (Deque):
1. Use deque for efficient access to both ends
2. Alternate between removing from front and back
3. Rebuild connections as we go

Approach 6 (Iterative with Length):
Similar to approach 1 but calculates length explicitly instead of using fast/slow pointers.

Approach Comparison:

1. Find Middle + Reverse + Merge (Best):
   - Time: O(n), Space: O(1)
   - Optimal solution
   - Three clear phases

2. Stack-based:
   - Time: O(n), Space: O(n)
   - Intuitive but uses extra space
   - Good for understanding

3. Array conversion:
   - Time: O(n), Space: O(n)
   - Simple but inefficient
   - Easy to implement

4. Recursive:
   - Time: O(n), Space: O(n)
   - Elegant but complex
   - Risk of stack overflow

5. Deque-based:
   - Time: O(n), Space: O(n)
   - Clean implementation
   - Uses standard data structure

6. Iterative with Length:
   - Time: O(n), Space: O(1)
   - Alternative to approach 1
   - Explicit length calculation

Key Insights:
1. Problem requires accessing both ends efficiently
2. Reversing second half enables easy merging
3. Finding middle is crucial for splitting
4. In-place solution is possible and optimal

Edge Cases:
- Empty list: no change needed
- Single node: no change needed
- Two nodes: 1->2 becomes 1->2 
(no change)
- Three nodes: 1->2->3 becomes 1->3->2
- Even vs odd length: algorithm handles both

Common Mistakes:
1. Not terminating the reordered list properly
2. Losing references during pointer manipulation
3. Incorrect middle finding for even-length lists
4. Not handling edge cases (empty, single node)
5. Creating cycles in the reordered list

Optimization Techniques:
1. Use slow/fast pointers to find middle in one pass
2. In-place reversal to save space
3. Careful pointer manipulation to avoid extra traversals
4. Early termination for small lists

Visual Representation:
Original: 1 -> 2 -> 3 -> 4 -> 5

Step 1 (Find middle):
1 -> 2 -> 3 -> 4 -> 5
          ^
        middle

Step 2 (Split and reverse):
First half:  1 -> 2 -> 3
Second half: 5 -> 4

Step 3 (Merge alternately):
1 -> 5 -> 2 -> 4 -> 3

Applications:
- Data structure manipulation
- Algorithm interview questions
- List processing in functional programming
- Memory-efficient data reorganization

Follow-up Questions:
1. Reorder in different patterns (e.g., L0→L2→L4→...→L1→L3→L5→...)
2. Reorder only part of the list
3. Reorder multiple lists simultaneously
4. Maintain original list while creating reordered copy

Performance Analysis:
- Optimal time: O(n) - must visit each node at least once
- Optimal space: O(1) - in-place manipulation
- Approach 1 achieves both optimal time and space
- Other approaches trade space for simplicity

Testing Strategy:
- Empty list: [] -> []
- Single node: [1] -> [1]
- Two nodes: [1,2] -> [1,2]
- Odd length: [1,2,3] -> [1,3,2]
- Even length: [1,2,3,4] -> [1,4,2,3]
- Longer list: [1,2,3,4,5,6] -> [1,6,2,5,3,4]
*/
```

That completes all 6 Linked List problems! Each solution includes:

1. **Multiple approaches** from basic to optimal
2. **Detailed algorithm explanations** with step-by-step examples
3. **Time and space complexity analysis** for each approach
4. **Key insights** and problem-solving strategies
5. **Edge case handling** and common mistakes
6. **Visual representations** and trace examples

The linked list problems covered important patterns like:
- **Two Pointers** (Fast/Slow for cycle detection, gap for nth from end)
- **List Reversal** (Iterative and recursive approaches)
- **List Merging** (Two lists, k lists with different strategies)
- **List Manipulation** (Reordering, removing nodes)
- **Space-Time Tradeoffs** (In-place vs extra space solutions)

----------------------------------------------------------
