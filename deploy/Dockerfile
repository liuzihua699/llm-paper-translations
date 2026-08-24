# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS builder
WORKDIR /site

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.29-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder --chmod=0555 /site/dist/ /usr/share/nginx/html/

EXPOSE 80
