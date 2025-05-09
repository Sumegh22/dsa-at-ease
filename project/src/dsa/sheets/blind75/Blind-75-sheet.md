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

------------------------------------
### 7. 