// Online Javascript Editor for free
// Write, Edit and Run your Javascript code using JS Online Compiler


const divmod = (x,y) => [Math.floor(x / y), x % y];

const remainder = []       // declare remainder array
let quocient = Infinity    // declare quocient array 
// This is called as recursion 
function remainderFinder(num, remainder) {
    const [val1,val2] = divmod(num,3)
    quocient = val1
    remainder.push(val2)
    if (quocient == 0) {
        return remainder
    }
    return remainderFinder(quocient,remainder)
 }



function series(num, pow=0,final_arr=[]) {
    const remainder = []  
    const sum = final_arr.reduce((a, b) => a + b, 0)
    if (sum === num){
        return final_arr
    }
    const diff = num - Math.pow(3,pow)
    const new_remainder = remainderFinder(diff,remainder)
    if (!new_remainder.includes(2)){
        final_arr.push(Math.pow(3,pow))
    }
    pow = pow + 1
    // final_arr.push(Math.pow(3,pow))
    return series(num,pow,final_arr)
    
}

// remainder finder
//In order to implement divmod() in javascript , we can use the build-in


//




    

// const divmod = (x, y) => [Math.floor(x / y), x % y];
// [[2,2][0,3][1,0]]
console.log(divmod(8,3))
console.log(divmod(3,8))
console.log(divmod(5,5))
console.log("remainder",remainderFinder(21 ,remainder))
console.log("series",series(91))
// console.log("remaiderarray-->", remainder)
