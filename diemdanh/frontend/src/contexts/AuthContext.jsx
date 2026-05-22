import { createContext, useContext, useMemo, useState } from "react";
import { api } from "../services/api";

const AuthContext = createContext(null);

function getInitialSession() {
  const token = sessionStorage.getItem("token");
  return {
    token,
    role: token ? sessionStorage.getItem("role") || "teacher" : "teacher",
    profileId: token ? sessionStorage.getItem("profileId") || "" : "",
    studentId: token ? sessionStorage.getItem("studentId") || "" : "",
    fullName: token ? sessionStorage.getItem("fullName") || "" : "",
  };
}

export function AuthProvider({ children }) {
  const [session, setSession] = useState(getInitialSession);

  function persistSession(next) {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("profileId");
    localStorage.removeItem("studentId");
    localStorage.removeItem("fullName");
    sessionStorage.setItem("token", next.token);
    sessionStorage.setItem("role", next.role);
    sessionStorage.setItem("profileId", next.profileId || "");
    sessionStorage.setItem("studentId", next.studentId || "");
    sessionStorage.setItem("fullName", next.fullName || "");
    setSession(next);
  }

  async function login(identifier, password, role = "teacher") {
    const loginId = role === "student" ? identifier.trim().toUpperCase() : identifier.trim();
    const { data } = await api.post("/auth/login", { email: loginId, password });
    persistSession({
      token: data.access_token,
      role: data.role || role,
      profileId: data.profile_id || loginId,
      studentId: data.student_id ? String(data.student_id) : "",
      fullName: data.full_name || "",
    });
  }

  function completeLogin(token, role, profileId, studentId = "", fullName = "") {
    persistSession({ token, role, profileId, studentId: String(studentId || ""), fullName });
  }

  function logout() {
    [localStorage, sessionStorage].forEach((storage) => {
      storage.removeItem("token");
      storage.removeItem("role");
      storage.removeItem("profileId");
      storage.removeItem("studentId");
      storage.removeItem("fullName");
    });
    setSession({ token: null, role: "teacher", profileId: "", studentId: "", fullName: "" });
  }

  const value = useMemo(() => ({
    token: session.token,
    role: session.role,
    profileId: session.profileId,
    studentId: session.studentId,
    fullName: session.fullName,
    login,
    completeLogin,
    logout,
    isAuthed: !!session.token,
  }), [session]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
