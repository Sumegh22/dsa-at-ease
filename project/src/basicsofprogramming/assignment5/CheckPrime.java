class CheckPrime {
    public boolean checkPrime(int num) {
        //your code goes here
        if(num<=1){
            return false;
        }
        return primeCheck(num, 2);
    }
    boolean primeCheck(int n, int i){
        if(i > Math.sqrt(n)){
            return true;
        } 
        if(n%i ==0){
            return false;
        }
        return primeCheck(n, i+1);

    }

  public static void main (String[] args){
    System.out.println(checkPrime(15));
  }
                           
}
