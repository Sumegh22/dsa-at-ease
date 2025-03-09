package project.src.dsa.aaBasicConcepts.basicArray;

public class OddNUmbersInArray {
    public int countOddNumbers(int[] nums) {
        int count = 0;
        for (int num : nums) {
            if (num % 2 != 0) {
                count++;
            }
        }
        return count;
    }

    public static void main(String[] args) {
        OddNUmbersInArray sol = new OddNUmbersInArray();
        int[] nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        System.out.println("Count of odd numbers in array is: " + sol.countOddNumbers(nums));
    }

}
