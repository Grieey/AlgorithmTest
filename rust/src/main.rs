use std::cmp;

fn main() {
    let mut nums = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4];
    let result = remove_duplicates(&mut nums);
    println!("结果是{}", result);

    let prices = vec![7, 1, 5, 3, 6, 4];
    let sell = max_profit(&prices);
    println!("结果是{}", sell);
}

pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    match nums.len() {
        0 => 0,
        1 => 1,
        n => {
            let mut index = 0;
            for i in 1..n {
                if nums[index] != nums[i] {
                    index += 1;
                    nums[index] = nums[i];
                }
            }
            (index + 1) as i32
        }
    }
}

pub fn max_profit(prices: &Vec<i32>) -> i32 {
    if prices.len() <= 1 {
        return 0;
    }

    let mut buy = -prices[0];
    let mut sell = 0;

    for price in prices {
        buy = cmp::max(buy, sell - price);
        sell = cmp::max(sell, buy + price);
    }

    sell
}

pub fn max(num1: i32, num2: i32) -> i32 {
    return if num1 > num2 {
        num1
    } else {
        num2
    };
}
