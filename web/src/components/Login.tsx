// The sign-in screen shown when the server runs with user accounts (`aime web
// --auth`): a name and a password, nothing else. A successful sign-in stores the
// session token, which api.ts attaches to every request and download link.
import { useState } from "react";

import { api, type AuthUser } from "../api";

export function Login({ onSignedIn }: { onSignedIn: (user: AuthUser) => void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !password || busy) return;
    setBusy(true);
    setError("");
    try {
      const { token, user } = await api.login(name.trim(), password);
      api.setToken(token);
      onSignedIn(user);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login-veil" role="dialog" aria-label="Sign in" data-testid="login">
      <form className="login-card panel" onSubmit={submit}>
        <div className="brand login-brand">AI <span>Made Easy</span></div>
        <h2>Sign in</h2>
        <p className="muted">This server keeps accounts. Ask an admin for one
          (<code>aime users add</code>).</p>
        <label htmlFor="login-name">Name</label>
        <input id="login-name" value={name} autoComplete="username" autoFocus
               onChange={(e) => setName(e.target.value)} />
        <label htmlFor="login-password">Password</label>
        <input id="login-password" type="password" value={password}
               autoComplete="current-password"
               onChange={(e) => setPassword(e.target.value)} />
        {error && <div className="login-error" role="alert">{error}</div>}
        <button className="primary" type="submit" disabled={busy || !name || !password}
                data-testid="login-submit">
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
