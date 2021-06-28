fn main() {
    let mut nums = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4];
    let result = remove_duplicates(&mut nums);
    println!("结果是{}", result);
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
