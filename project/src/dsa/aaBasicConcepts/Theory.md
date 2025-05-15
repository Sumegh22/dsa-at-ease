# DSA Introduction and Theory 


## Arrays :
1. Arrays represent one of the most fundamental data structures in programming, designed to store a collection of elements in a linear format. 
2. Each element in an array is of the same data type, which can include integers, floats, or doubles, among others.
3. int[] arr = new int[4]    arr = [0,1,2,3] ;

------------------------

## Hashing : 
1. Hashing is a technique used in computer science to quickly find, store, and manage data. It works by taking an input, like a number or a string, and converting it into a fixed-size value called a hash. This hash then points to where the data is stored in a structure called a hash table. The main goal of hashing is to make data retrieval fast, even when dealing with large amounts of information. 
2. Hashing is widely used in various applications, such as searching databases, managing passwords, and speeding up data lookups in many types of software.

## String : 
1. A string is a sequence of characters that are often used to represent text. Strings are a fundamental data type in almost every programming language, providing a mechanism to store and manipulate text efficiently. 
2. The concept of strings is universally applicable, yet their implementation and behavior can vary significantly across different languages.
3. Java: String str = "Hello";
------------------------

## Sorting: 
* A sorted array in non-decreasing order is an array where each element is greater than or equal to all previous elements in the array.

**1. Selection Sort :**
* Intuition : 
   The selection sort algorithm sorts an array by repeatedly finding the minimum element from the unsorted part and putting it at the beginning. The largest element will end up at the last index of the array.

**2. Bubble Sort:**
* Intuition : 
   The bubble sort algorithm sorts an array by repeatedly swapping adjacent elements if they are in the wrong order. The largest elements "bubble" to the end of the array with each pass.

**3. Insertion Sorting**
* Intuition:
  Insertion sort builds a sorted array one element at a time by repeatedly picking the next element and inserting it into its correct position within the already sorted part of the array.

**4. Merge Sort**
* Intuition:
Merge Sort is a powerful sorting algorithm that follows the divide-and-conquer approach. The array is divided into two equal halves until each sub-array contains only one element. Each pair of smaller sorted arrays is then merged into a larger sorted array.

**5. Quick Sort**
* Intuition:
  Quick Sort is a divide-and-conquer algorithm like Merge Sort. However, unlike Merge Sort, Quick Sort does not use an extra array for sorting (though it uses an auxiliary stack space). This makes Quick Sort slightly better than Merge Sort from a space perspective.

------------------------
