package project.src.dsa.array.easy;

public class SecondLargest {
    public static void main(String[] args) {
        int[] input = {12, 35, 1, 10, 34, 1};
        int n = input.length;

        int mx = Integer.MIN_VALUE;
        int sm = Integer.MIN_VALUE;

        for (int i = 0; i < n; i++) {
            if (input[i] > mx) {
                sm = mx;
                mx = input[i];
            } else if (input[i] > sm && input[i] < mx) {
                sm = input[i];
            }
        }

        if (sm == Integer.MIN_VALUE) {
            System.out.println("No second largest element found");
        } else {
            System.out.println("Second Largest Element: " + sm);
        }
    }
}

