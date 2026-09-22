--Exercise1

DROP TABLE IF EXISTS df_employee;

CREATE TABLE df_employee AS
SELECT 
    employee_id || '_' || CAST(date AS TEXT) AS id,
    DATE(date) AS month_year,
    employee_id,
    employee_name,
    gender,
    age,
    salary,
    function_group,
    company_name,
    company_city,
    company_state,
    company_type,
    const_site_category
FROM emp_dataset;


DROP TABLE IF EXISTS emp_dataset;

CREATE TEMP TABLE emp_dataset AS
SELECT
    s.employee_id,
    s.employee_name,
    s.date,
    e.gen_m_f,
    e.age,
    s.salary,
    f.function_group,
    c.company_name,
    c.company_city,
    c.company_state,
    c.company_type,
    c.const_site_categ
FROM salaries s

LEFT JOIN employees e
    ON s.employee_id = e.employee_code_em
    AND s.comp_code = e.comp_code_emp

LEFT JOIN functions f
    ON s.func_code = f.function_code

LEFT JOIN companies c
    ON s.comp_name = c.company_name;

DROP TABLE IF EXISTS df_employee;

CREATE TABLE df_employee AS
SELECT
    employee_id || '_' || CAST(date AS TEXT) AS id,
    DATE(date) AS month_year,
    employee_id,
    employee_name,
    gen_m_f AS gender,
    age,
    salary,
    function_group,
    company_name,
    company_city,
    company_state,
    company_type,
    const_site_categ AS const_site_category
FROM emp_dataset;


DROP TABLE IF EXISTS df_employee;

CREATE TABLE df_employee AS
SELECT
    employee_id || '_' || CAST(date AS TEXT) AS id,

    SUBSTR(date, 7, 4) || '-' ||
    SUBSTR(date, 4, 2) || '-' ||
    SUBSTR(date, 1, 2) AS month_year,

    employee_id,
    employee_name,
    gen_m_f AS gender,
    age,
    salary,
    function_group,
    company_name,
    company_city,
    company_state,
    company_type,
    const_site_categ AS const_site_category

FROM emp_dataset;


--Exercise2

DROP TABLE IF EXISTS emp_dataset;

CREATE TEMP TABLE emp_dataset AS
SELECT
    s.employee_id,
    s.employee_name,
    s.date,
    e.gen_m_f,
    e.age,
    s.salary,
    f.function_group,
    c.company_name,
    c.company_city,
    c.company_state,
    c.company_type,
    c.const_site_categ
FROM salaries s

LEFT JOIN employees e
    ON s.employee_id = e.employee_code_em

LEFT JOIN functions f
    ON s.func_code = f.function_code

LEFT JOIN companies c
    ON s.comp_name = c.company_name;

DROP TABLE IF EXISTS df_employee;

CREATE TABLE df_employee AS
SELECT
    employee_id || '_' || CAST(date AS TEXT) AS id,

    SUBSTR(date, 7, 4) || '-' ||
    SUBSTR(date, 4, 2) || '-' ||
    SUBSTR(date, 1, 2) AS month_year,

    employee_id,
    employee_name,
    gen_m_f AS gender,
    age,
    salary,
    function_group,
    company_name,
    company_city,
    company_state,
    company_type,
    const_site_categ AS const_site_category
FROM emp_dataset;

UPDATE df_employee
SET
    id = TRIM(id),
    month_year = TRIM(month_year),
    employee_name = TRIM(employee_name),
    gender = TRIM(gender),
    salary = TRIM(salary),
    function_group = TRIM(function_group),
    company_name = TRIM(company_name),
    company_city = TRIM(company_city),
    company_state = TRIM(company_state),
    company_type = TRIM(company_type),
    const_site_category = TRIM(const_site_category);

SELECT *
FROM df_employee
WHERE id IS NULL OR id = ''
   OR month_year IS NULL OR month_year = ''
   OR employee_id IS NULL
   OR employee_name IS NULL OR employee_name = ''
   OR gender IS NULL OR gender = ''
   OR age IS NULL
   OR salary IS NULL OR salary = ''
   OR function_group IS NULL OR function_group = ''
   OR company_name IS NULL OR company_name = ''
   OR company_city IS NULL OR company_city = ''
   OR company_state IS NULL OR company_state = ''
   OR company_type IS NULL OR company_type = '';

DELETE FROM df_employee
WHERE salary IS NULL
   OR TRIM(salary) = '';


SELECT
    company_name,
    COUNT(DISTINCT employee_id) AS employee_count
FROM df_employee
WHERE month_year = (
    SELECT MAX(month_year)
    FROM df_employee
)
GROUP BY company_name
ORDER BY employee_count DESC;

SELECT
    company_city,
    COUNT(DISTINCT employee_id) AS employee_count,
    ROUND(
        COUNT(DISTINCT employee_id) * 100.0 /
        (
            SELECT COUNT(DISTINCT employee_id)
            FROM df_employee
            WHERE month_year = (
                SELECT MAX(month_year)
                FROM df_employee
            )
        ),
        2
    ) AS percentage
FROM df_employee
WHERE month_year = (
    SELECT MAX(month_year)
    FROM df_employee
)
GROUP BY company_city
ORDER BY employee_count DESC;

SELECT
    month_year,
    COUNT(DISTINCT employee_id) AS employee_count
FROM df_employee
GROUP BY month_year
ORDER BY month_year;

SELECT
    ROUND(AVG(employee_count), 2) AS avg_employees_per_month
FROM (
    SELECT
        month_year,
        COUNT(DISTINCT employee_id) AS employee_count
    FROM df_employee
    GROUP BY month_year
);



WITH monthly_counts AS (
    SELECT
        month_year,
        COUNT(DISTINCT employee_id) AS employee_count
    FROM df_employee
    GROUP BY month_year
)

SELECT
    'Minimum' AS count_type,
    month_year,
    employee_count
FROM monthly_counts
WHERE employee_count = (
    SELECT MIN(employee_count)
    FROM monthly_counts
)

UNION ALL

SELECT
    'Maximum',
    month_year,
    employee_count
FROM monthly_counts
WHERE employee_count = (
    SELECT MAX(employee_count)
    FROM monthly_counts
);

WITH monthly_function_counts AS (
    SELECT
        month_year,
        function_group,
        COUNT(DISTINCT employee_id) AS employee_count
    FROM df_employee
    GROUP BY month_year, function_group
)

SELECT
    function_group,
    ROUND(AVG(employee_count), 2) AS avg_monthly_employees
FROM monthly_function_counts
GROUP BY function_group
ORDER BY avg_monthly_employees DESC;


SELECT
    SUBSTR(month_year, 1, 4) AS year,
    ROUND(
        AVG(
            CAST(REPLACE(salary, ',', '.') AS REAL)
        ),
        2
    ) AS average_salary
FROM df_employee
GROUP BY SUBSTR(month_year, 1, 4)
ORDER BY year;

