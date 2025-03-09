package project.src.dsa.aaBasicConcepts.basicArray;

public class SumOfElements {
    public int sumOfElements(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        return sum;
    }

    public static void main(String[] args) {
        SumOfElements sol = new SumOfElements();
        int[] nums = {1, 2, 3, 4, 5};
        System.out.println("Sum of elements in array is: " + sol.sumOfElements(nums));
    }

}
