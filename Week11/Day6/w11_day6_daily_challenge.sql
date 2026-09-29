-- DROP TABLE IF EXISTS employees;

-- CREATE TABLE employees (
--     employee_id INT PRIMARY KEY,
--     employee_name VARCHAR(50),
--     salary DECIMAL(10, 2),
--     hire_date VARCHAR(20),
--     department VARCHAR(50)
-- );

-- INSERT INTO employees
-- (employee_id, employee_name, salary, hire_date, department)
-- VALUES
-- (1, 'Amy West', 60000.00, '2021-01-15', 'HR'),
-- (2, 'Ivy Lee', 75000.50, '2020-05-22', 'Sales'),
-- (3, 'joe smith', 80000.75, '2019-08-10', 'Marketing'),
-- (4, 'John White', 90000.00, '2020-11-05', 'Finance'),
-- (5, 'Jane Hill', 55000.25, '2022-02-28', 'IT'),
-- (6, 'Dave West', 72000.00, '2020-03-12', 'Marketing'),
-- (7, 'Fanny Lee', 85000.50, '2018-06-25', 'Sales'),
-- (8, 'Amy Smith', 95000.25, '2019-11-30', 'Finance'),
-- (9, 'Ivy Hill', 62000.75, '2021-07-18', 'IT'),
-- (10, 'Joe White', 78000.00, '2022-04-05', 'Marketing'),
-- (11, 'John Lee', 68000.50, '2018-12-10', 'HR'),
-- (12, 'Jane West', 89000.25, '2017-09-15', 'Sales'),
-- (13, 'Dave Smith', 60000.75, '2022-01-08', NULL),
-- (14, 'Fanny White', 72000.00, '2019-04-22', 'IT'),
-- (15, 'Amy Hill', 84000.50, '2020-08-17', 'Marketing'),
-- (16, 'Ivy West', 92000.25, '2021-02-03', 'Finance'),
-- (17, 'Joe Lee', 58000.75, '2018-05-28', 'IT'),
-- (18, 'John Smith', 77000.00, '2019-10-10', 'HR'),
-- (19, 'Jane Hill', 81000.50, '2022-03-15', 'Sales'),
-- (20, 'Dave White', 70000.25, '2017-12-20', 'Marketing');



-- 1. Identify and handle missing values
SELECT *
FROM employees
WHERE employee_id IS NULL
   OR employee_name IS NULL
   OR salary IS NULL
   OR hire_date IS NULL
   OR department IS NULL;

UPDATE employees
SET department = 'Unknown'
WHERE department IS NULL
   OR TRIM(department) = '';   


-- 2. Check for duplicate rows
SELECT
    employee_name,
    salary,
    hire_date,
    department,
    COUNT(*) AS duplicate_count
FROM employees
GROUP BY
    employee_name,
    salary,
    hire_date,
    department
HAVING COUNT(*) > 1;


-- 3. Correct inconsistent formatting
-- UPDATE employees
-- SET
--     employee_name = TRIM(employee_name),
--     hire_date = TRIM(hire_date),
--     department = TRIM(department);

-- UPDATE employees
-- SET employee_name = INITCAP(employee_name);	

-- UPDATE employees
-- SET department =
--     CASE
--         WHEN LOWER(department) = 'hr' THEN 'HR'
--         WHEN LOWER(department) = 'it' THEN 'IT'
--         WHEN LOWER(department) = 'sales' THEN 'Sales'
--         WHEN LOWER(department) = 'marketing' THEN 'Marketing'
--         WHEN LOWER(department) = 'finance' THEN 'Finance'
--         WHEN LOWER(department) = 'unknown' THEN 'Unknown'
--         ELSE department
--     END;


-- 4. Convert hire_date to a proper DATE

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'employees';

ALTER TABLE employees
ALTER COLUMN hire_date TYPE DATE
USING hire_date::DATE;

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'employees';


-- 5. Detect salary outliers
-- SELECT
--     PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS q1,
--     PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS q3
-- FROM employees;

-- WITH quartiles AS (
--     SELECT
--         PERCENTILE_CONT(0.25)
--             WITHIN GROUP (ORDER BY salary) AS q1,
--         PERCENTILE_CONT(0.75)
--             WITHIN GROUP (ORDER BY salary) AS q3
--     FROM employees
-- ),
-- limits AS (
--     SELECT
--         q1,
--         q3,
--         q1 - 1.5 * (q3 - q1) AS lower_limit,
--         q3 + 1.5 * (q3 - q1) AS upper_limit
--     FROM quartiles
-- )
-- SELECT
--     e.*
-- FROM employees e
-- CROSS JOIN limits l
-- WHERE e.salary < l.lower_limit
--    OR e.salary > l.upper_limit;
SELECT
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    ROUND(AVG(salary), 2) AS avg_salary
FROM employees;


-- 6. Final standardization

-- ALTER TABLE employees
-- ALTER COLUMN employee_name SET NOT NULL;

-- ALTER TABLE employees
-- ALTER COLUMN salary SET NOT NULL;

-- ALTER TABLE employees
-- ALTER COLUMN hire_date SET NOT NULL;

-- ALTER TABLE employees
-- ALTER COLUMN department SET NOT NULL;

SELECT *
FROM employees
ORDER BY employee_id;

SELECT *
FROM employees
WHERE employee_name IS NULL
   OR TRIM(employee_name) = ''
   OR salary IS NULL
   OR hire_date IS NULL
   OR department IS NULL
   OR TRIM(department) = '';


-- Final results for this particular dataset
-- After cleaning, we should have:
-- - missing department → Unknown
-- - joe smith → Joe Smith
-- - whitespace removed
-- - no true duplicate rows found
-- - hire_date changed from VARCHAR to DATE
-- - salary retained as DECIMAL(10,2)
-- - department labels standardized
-- - no obvious salary outliers requiring removal   