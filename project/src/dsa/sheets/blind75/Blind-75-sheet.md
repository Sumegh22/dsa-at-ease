#### Solving leet code blind 75

# Arrays Problem

### 1. Two Sum
     * The optimal approach for this code uses time complexity of O(n)

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



* Example 1:

    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.

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
### 7. Find min in a rotated sorted Array 

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
                if(target>nums[right]|| target<nums[mid]) right = mid-1;
                else left= mid+1;
            }
        }
        return -1;
    }
    
}
```

