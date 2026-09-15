-- Part I.
-- -- 1. Create the tables

-- CREATE TABLE customer (
--     id SERIAL PRIMARY KEY,
--     first_name VARCHAR(50),
--     last_name VARCHAR(50) NOT NULL
-- );

-- CREATE TABLE customer_profile (
--     id SERIAL PRIMARY KEY,
--     "isLoggedIn" BOOLEAN DEFAULT FALSE,
--     customer_id INTEGER UNIQUE,
    
--     FOREIGN KEY (customer_id)
--         REFERENCES customer(id)
--         ON DELETE CASCADE
-- );

-- -- 2.
-- INSERT INTO customer (first_name, last_name)
-- VALUES
--     ('John', 'Doe'),
--     ('Jerome', 'Lalu'),
--     ('Lea', 'Rive');
	
-- -- 3.
-- INSERT INTO customer_profile ("isLoggedIn", customer_id)
-- VALUES (
--     TRUE,
--     (SELECT id
--      FROM customer
--      WHERE first_name = 'John'
--        AND last_name = 'Doe')
-- );

-- INSERT INTO customer_profile ("isLoggedIn", customer_id)
-- VALUES (
--     FALSE,
--     (SELECT id
--      FROM customer
--      WHERE first_name = 'Jerome'
--        AND last_name = 'Lalu')
-- );

-- 4.
SELECT customer.first_name
FROM customer
JOIN customer_profile
    ON customer.id = customer_profile.customer_id
WHERE customer_profile."isLoggedIn" = TRUE;

SELECT
    customer.first_name,
    customer_profile."isLoggedIn"
FROM customer
LEFT JOIN customer_profile
    ON customer.id = customer_profile.customer_id;

SELECT COUNT(*)
FROM customer
JOIN customer_profile
    ON customer.id = customer_profile.customer_id
WHERE customer_profile."isLoggedIn" = FALSE;


-- Part II.
-- 1.
-- CREATE TABLE book (
--     book_id SERIAL PRIMARY KEY,
--     title VARCHAR(50) NOT NULL,
--     author VARCHAR(50) NOT NULL
-- );

-- 2.
-- INSERT INTO Book (title, author)
-- VALUES
--     ('Alice In Wonderland', 'Lewis Carroll'),
--     ('Harry Potter', 'J.K Rowling'),
--     ('To kill a mockingbird', 'Harper Lee');

-- 3.
-- CREATE TABLE Student (
--     student_id SERIAL PRIMARY KEY,
--     name VARCHAR(50) NOT NULL UNIQUE,
--     age INTEGER CHECK (age <= 15)
-- );

-- 4.
-- INSERT INTO Student (name, age)
-- VALUES
--     ('John', 12),
--     ('Lera', 11),
--     ('Patrick', 10),
--     ('Bob', 14);

-- 5.
-- CREATE TABLE Library (
--     book_fk_id INTEGER,
--     student_fk_id INTEGER,
--     borrowed_date DATE,

--     PRIMARY KEY (book_fk_id, student_fk_id),

--     FOREIGN KEY (book_fk_id)
--         REFERENCES Book(book_id)
--         ON DELETE CASCADE
--         ON UPDATE CASCADE,

--     FOREIGN KEY (student_fk_id)
--         REFERENCES Student(student_id)
--         ON DELETE CASCADE
--         ON UPDATE CASCADE
-- );

-- 6.
-- INSERT INTO Library (book_fk_id, student_fk_id, borrowed_date)
-- VALUES
-- (
--     (SELECT book_id FROM Book WHERE title = 'Alice In Wonderland'),
--     (SELECT student_id FROM Student WHERE name = 'John'),
--     '2022-02-15'
-- ),
-- (
--     (SELECT book_id FROM Book WHERE title = 'To kill a mockingbird'),
--     (SELECT student_id FROM Student WHERE name = 'Bob'),
--     '2021-03-03'
-- ),
-- (
--     (SELECT book_id FROM Book WHERE title = 'Alice In Wonderland'),
--     (SELECT student_id FROM Student WHERE name = 'Lera'),
--     '2021-05-23'
-- ),
-- (
--     (SELECT book_id FROM Book WHERE title = 'Harry Potter'),
--     (SELECT student_id FROM Student WHERE name = 'Bob'),
--     '2021-08-12'
-- );

-- 7.
-- 7.1. Select all columns from the junction table
SELECT *
FROM Library;


-- 7.2. Student name + borrowed book title
SELECT
    Student.name,
    Book.title
FROM Library
JOIN Student
    ON Library.student_fk_id = Student.student_id
JOIN Book
    ON Library.book_fk_id = Book.book_id;

-- 7.3. Average age of students who borrowed Alice In Wonderland
SELECT AVG(Student.age)
FROM Library
JOIN Student
    ON Library.student_fk_id = Student.student_id
JOIN Book
    ON Library.book_fk_id = Book.book_id
WHERE Book.title = 'Alice In Wonderland';

-- 7.4.
-- DELETE FROM Student
-- WHERE name = 'Bob';

SELECT *
FROM Library;