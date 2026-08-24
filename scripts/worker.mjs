export default {
  fetch(request, environment) {
    const url = new URL(request.url);
    if (url.pathname === "/") {
      url.pathname = "/index.html";
      return environment.ASSETS.fetch(new Request(url, request));
    }
    return environment.ASSETS.fetch(request);
  },
};
