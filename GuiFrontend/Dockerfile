FROM node:18.20-slim as BUILDER

WORKDIR /app
#RUN npm install typescript -g
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

#CMD npm run preview

FROM nginx:1.16.0-alpine

COPY --from=BUILDER /app/dist /var/www/html/
RUN rm /etc/nginx/conf.d/default.conf
COPY ./nginx.conf /etc/nginx/conf.d

CMD ["nginx","-g","daemon off;"]

