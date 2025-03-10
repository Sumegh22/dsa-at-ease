package project.src.dsa.aaBasicConcepts.basicArray;

public class ReverseAnArray {
    // Function to reverse array using two pointers

    public void reverse(int[] arr, int n) {
        int p1 = 0, p2 = n - 1;
        /* Swap elements pointed by p1 and  p2 until they meet in the middle */
        while (p1 < p2) {
            int tmp = arr[p1];
            arr[p1] = arr[p2];
            arr[p2] = tmp;
            p1++;
            p2--;
        }
        return ;
    }


    // Function to print array
    static void printArray(int arr[], int n) {
        for (int i = 0; i < n; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }


    public static void main(String[] args) {
        int n = 5;
        int[] arr = {5, 4, 3, 2, 1};

        ReverseAnArray solution = new ReverseAnArray();
        System.out.print("Original array: ");
        printArray(arr, n);

        solution.reverse(arr, n);
        System.out.print("Reversed array: ");
        printArray(arr, n);
    }
}

