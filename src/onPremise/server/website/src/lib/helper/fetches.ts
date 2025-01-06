const SERVER_URL = import.meta.env.VITE_SERVER_URL;

export async function update_cloud_url(url: any): Promise<Response> {
  let response = await fetch(`${SERVER_URL}/api/update-cloud-url?cloud_url=${url}`, {
    method: 'POST',
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  });

  return response
}

export async function getCompany() {
  let res = await fetch(`${SERVER_URL}/api/get-company`, {
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  });
  return res;
}