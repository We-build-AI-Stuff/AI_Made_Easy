// Admin's user management: list accounts, add one, change a password or a role,
// disable or remove — the same things `aime users` does, from the browser.
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { Modal } from "./Modal";

interface Row { name: string; role: string; disabled: boolean }

export function UsersModal({ me, onClose }: { me: string; onClose: () => void }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState("member");

  const refresh = useCallback(async () => {
    try {
      setRows((await api.users()).users);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  const act = async (what: string, fn: () => Promise<unknown>) => {
    try {
      await fn();
      setNotice(what);
      setError("");
      await refresh();
    } catch (e) {
      setError((e as Error).message);
      setNotice("");
    }
  };

  return (
    <Modal title="Users" onClose={onClose} wide>
      <p className="muted">Members work on their own projects; admins manage users and see every
        project. <code>aime users</code> does the same from a terminal.</p>
      {error && <div className="login-error" role="alert">{error}</div>}
      {notice && <div className="toast-line" data-testid="users-notice">{notice}</div>}
      <table>
        <thead>
          <tr><th>Name</th><th>Role</th><th>Status</th><th /></tr>
        </thead>
        <tbody>
          {rows.map((u) => (
            <tr key={u.name} data-testid={`user-${u.name}`}>
              <td>{u.name}{u.name === me ? <span className="muted"> (you)</span> : null}</td>
              <td>
                <select value={u.role} disabled={u.name === me}
                        onChange={(e) => act(`changed ${u.name}'s role`,
                          () => api.patchUser(u.name, { role: e.target.value }))}>
                  <option value="member">member</option>
                  <option value="admin">admin</option>
                </select>
              </td>
              <td>{u.disabled ? "disabled" : "active"}</td>
              <td className="row-actions">
                <button onClick={() => {
                  const pw = window.prompt(`New password for ${u.name}`);
                  if (pw) void act(`changed ${u.name}'s password`,
                    () => api.patchUser(u.name, { password: pw }));
                }}>Password…</button>
                <button onClick={() => act(u.disabled ? `enabled ${u.name}` : `disabled ${u.name}`,
                  () => api.patchUser(u.name, { disabled: !u.disabled }))}>
                  {u.disabled ? "Enable" : "Disable"}
                </button>
                <button className="danger" disabled={u.name === me}
                        onClick={() => act(`removed ${u.name}`, () => api.removeUser(u.name))}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="user-add">
        {adding ? (
          <>
            <input placeholder="name" value={newName} aria-label="New user's name"
                   onChange={(e) => setNewName(e.target.value)} />
            <input placeholder="password" type="password" value={newPassword}
                   aria-label="New user's password"
                   onChange={(e) => setNewPassword(e.target.value)} />
            <select value={newRole} aria-label="New user's role"
                    onChange={(e) => setNewRole(e.target.value)}>
              <option value="member">member</option>
              <option value="admin">admin</option>
            </select>
            <button className="primary" disabled={!newName.trim() || !newPassword}
                    onClick={() => act(`added ${newName.trim()}`, async () => {
                      await api.addUser(newName.trim(), newPassword, newRole);
                      setAdding(false); setNewName(""); setNewPassword("");
                    })}>Add</button>
            <button className="ghost" onClick={() => setAdding(false)}>Cancel</button>
          </>
        ) : (
          <button onClick={() => setAdding(true)} data-testid="add-user">Add a user…</button>
        )}
      </div>
    </Modal>
  );
}
