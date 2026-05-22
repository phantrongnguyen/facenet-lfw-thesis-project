import axios from "axios";

export const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8010/api" });

api.interceptors.request.use((config)=>{
  const t=sessionStorage.getItem("token") || localStorage.getItem("token");
  if(t) config.headers.Authorization=`Bearer ${t}`;
  return config;
});

export async function upload(path, file, fieldName="file"){
  const fd=new FormData();
  fd.append(fieldName, file);
  const {data}=await api.post(path, fd, {headers:{"Content-Type":"multipart/form-data"}});
  return data;
}

export async function registerStudentWithFace({studentCode, fullName, password, faceImage}){
  const fd=new FormData();
  fd.append("student_code", studentCode);
  fd.append("full_name", fullName);
  fd.append("password", password);
  fd.append("face_image", faceImage);
  const {data}=await api.post("/auth/students/register", fd, {headers:{"Content-Type":"multipart/form-data"}});
  return data;
}
