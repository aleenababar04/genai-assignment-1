# Frontend image: build the React app with Node, serve the static files with nginx.
#
# Build context: the frontend/ folder (see docker-compose.yml at the repo root).

# ---- Stage 1: build -------------------------------------------------------
FROM node:24-alpine AS build
WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- Stage 2: serve -------------------------------------------------------
FROM nginx:alpine

# nginx serves the built app and forwards /api/ to the backend container.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
