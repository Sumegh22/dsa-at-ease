#### Solving leet code blind 75

# Arrays Problem

### 1. Two Sum
* The optimal approach for this code uses **time complexity of O(n)**

**Solution:**

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map= new HashMap<>();
        int n = nums.length;

        for(int i =0; i<n; i++){
            int diff = target-nums[i];
            if(map.containsKey(diff)){
                return new int[] {i, map.get(diff)};
            } else{
                map.put(nums[i], i);
            }
        }


        return new int[] {-1, -1} ;
    }
}

```

---------------------------

### 2. Best Time to Buy and Sell Stock 

* You are given an array prices where prices[i] is the price of a given stock on the ith day.
* You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
* Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

**Solution:**
```java

class Solution {
    public int maxProfit(int[] prices) {
        int profit = 0; 
        int buyPrice = prices[0];

        for(int i=0; i<prices.length; i++){
            if(prices[i]<buyPrice){
                buyPrice = prices[i];
            }
            profit = Math.max(profit, prices[i]- buyPrice);
        }



        return profit;
    }
}
```
------------------------------------

### 3. 217. Contains Duplicate
* Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.


**Solution:**
```java

class Solution {
    public boolean containsDuplicate(int[] nums) {

        Set<Integer> visited = new HashSet<>();
        
        for(int i=0; i<nums.length; i++){
            if(visited.contains(nums[i])){
                return true;
            } else{
                visited.add(nums[i]);
            }
        }
        return false;
    }
}
```
------------------------------------

### 4. Product of Array Except Self

* Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
* The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
* You must write an algorithm that runs in O(n) time and without using the division operation.


**Solution:** 
```java

class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length; 
        int prefix =1; 
        int postfix = 1; 
        int[] ans = new int[n];

        for(int i =0; i<n; i++){
            ans[i] = prefix; 
            prefix *= nums[i];
        }

        for(int i = n-1; i>=0; i--){
            ans[i] *= postfix;
            postfix*=nums[i];
        }
        return ans;
    }
}

```
------------------------------------

### 5. Leetcode : 53. Maximum Subarray ([here](https://leetcode.com/problems/maximum-subarray/))
* Given an integer array nums, find the subarray with the largest sum, and return its sum.



Example 1:

    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.

**Solution:**

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int max = nums[0]; 
        int current = 0;

        for(int i=0; i<nums.length; i++){
            if(current < 0){
                current = 0; 
            }

            current += nums[i];
            max = Math.max(current, max);
        }
        return max;
        
    }
}
```
------------------------------------

### 6. Leetcode: 152. Maximum Product Subarray ([here](https://leetcode.com/problems/maximum-product-subarray/description/)) 

* Given an integer array nums, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

Example 1:

    Input: nums = [2,3,-2,4]
    Output: 6
    Explanation: [2,3] has the largest product 6.

Example 2:

    Input: nums = [-2,0,-1]
    Output: 0
    Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

**Solution:**

```java

class Solution{
  static int maxProduct(int[] nums){
      if(nums.length == 1) return nums[0];
      
      int min = nums[0];
      int max = nums[0];
      
      for(int i=1; i<nums.length; i++){
          int curr = nums[i];
          
          int temp = Math.max(max, Math.max(min*curr, max*curr));
          min = Math.min(min, Math.min(min*curr, max*curr));
          max = temp;
      }
      return max;
  }
}

```

------------------------------------
### 7. Find min in a rotated sorted Array  ([here](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/))

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

```declarative
Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
```

```java

class Solution{
    static int findMinInRotatedSortedArray(int[] nums){
        if (nums.length ==1) return nums[0];
        
        int left = 0; 
        int right = nums.length-1;
        int min = nums[0];
        // 345120
        while(left<=right){
            if(nums[left]< nums[right]){
                min = Math.min(min, nums[left]);
            }
            int mid = (left+right)/2;
            min = Math.min(min, nums[mid]);
            if(nums[left]<=nums[mid]){
                left=mid+1;
            } else{
                right=mid-1;
            }
        }
        return min;
    }
}
```
----------------------------------------
### 8. Search in a rotated sorted Array ([here](https://leetcode.com/problems/search-in-rotated-sorted-array/description/))


* There is an integer array nums sorted in ascending order (with distinct values).

* Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

* Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

* You must write an algorithm with O(log n) runtime complexity.


```declarative
Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1
```
**Solution**

```java

class Solution{
    static int searchInRotatedSortedArray(int[] nums, int target){
        if(nums.length ==1){
            if (nums[0]== target) return 0;
            else return -1;
        }
        
        int left = 0; int right = nums.length-1;
        
        while(left<=right){
            int mid = (left+right)/2;
            if(target == nums[mid]) return mid;
            
            if(nums[left] <= nums[mid]){
                if(target<nums[left] || target>nums[mid]) left= mid+1;
                else right = mid-1;
            } else{
                if(target<nums[mid] || target>nums[right]) right = mid-1;
                else left= mid+1;
            }
        }
        return -1;
    }
    
}
```

----------------------------------

### 9. 2 Sum II 

Two Sum II - Input Array Is Sorted Medium

* Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

* Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

* The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.

Example 1:
```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
```

Example 2:
```
Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
```
Example 3:
```
Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
```

**Solution**

```java

class Solution {
    public int[] twoSum(int[] nums, int target) {
        int left = 0; 
        int right= nums.length-1;

        while(left<right){
            int sum = nums[left]+nums[right];

            if(sum>target){
                right-=1;
            }
            else if(sum < target) {
                left+=1;
            }
            else {
                return new int[] {left+1, right+1};
            }
        }
        return null;       
    }
}
```
--------------------------------------

### 10. 3Sum 

* Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

* Notice that the solution set must not contain duplicate triplets.



Example 1:
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation:
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
```

Example 2:
```
Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.
```
Example 3:
```
Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.
```
**Solution**

```java

class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        
        for (int i = 0; i<nums.length && nums[i] <=0; i++){
            if (i == 0 || nums[i] != nums[i-1]){
                twoSum2(nums, i, result);
            }
        }
        
        return result;
        
    }
    
    void twoSum2(int[] nums, int i, List<List<Integer>> result){
        int left = i+1;
        int right = nums.length - 1;
        
        while(left < right){
            int sum = nums[i] + nums[left] + nums[right];
            
            if(sum < 0){
                ++left;
            }
            else if (sum > 0){
                --right;
            }
            else{
                result.add(Arrays.asList(nums[i], nums[left++], nums[right--]));
                while(left < right && nums[left] == nums[left-1]){
                    ++left;
                }
            }
        }
    }
}

```
--------------------------------------

### 11. Container With Most Water 
    Medium
    Hint: 
* You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

* Find two lines that together with the x-axis form a container, such that the container contains the most water.

* Return the maximum amount of water a container can store.

* Notice that you may not slant the container.



Example 1:
    ![question_11.jpg](../quick-refreshers/question_11.jpg)

    Input: height = [1,8,6,2,5,4,8,3,7]
    Output: 49
    Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Example 2:

Input: height = [1,1]
Output: 1

**Solution**
```java

class Solution {
    public int maxArea(int[] height) {
        int max = 0;
        int left = 0;
        int right = height.length-1;

        while(left<right){
            int width = right - left;
            int area = (Math.min(height[left], height[right]) * width);
            max = Math.max(area, max);

            if(height[left]<= height[right]){
                left++;
            } else{
                right--;
            }
        }
        return max;
    }
}
```
-----------------------------

