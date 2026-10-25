import { redirect } from "next/navigation";

export default function Page() {
  // Landing on the root sends the user straight to the workspace Overview.
  redirect("/overview");
}
