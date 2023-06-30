create table user
(
    id       int unsigned auto_increment comment 'ID'
        primary key,
    username varchar(10)                  not null comment '用户名',
    password varchar(10) default '123456' not null comment '密码',
    constraint user_pk2
        unique (username)
)
    comment '用户信息表';
